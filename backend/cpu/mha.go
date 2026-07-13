package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized fused attention (§T599, profile-driven: the GPT cpu training step
// spent 45% of its samples in ref's interface-arithmetic MHA backward and 13%
// in its forward). Same semantics as backend/ref's kernels — GQA/MQA, causal,
// KV-cache offset (forward), ALiBi, sliding window, YaRN-folded scale — on
// contiguous typed slices. Forward parallelizes over (head, query row);
// backward over KV-HEAD GROUPS, whose dQ/dK/dV regions are disjoint (the rep
// query heads of a group share exactly that group's K/V columns), so there is
// no scatter contention and no atomics. All accumulation in f64 (§V10); dK/dV
// accumulate in f64 scratch and store once — more accurate than ref's
// through-the-tensor sums, inside the ulps parity budget (§T596's V9 note).
func mhaKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: mha wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: mha needs rank-2 [seq,dmodel]")
	}
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: mha dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: mha heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	kvDM := kvHeads * dk
	if k.Shape()[1] != kvDM || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("cpu: mha K/V must be [sk,%d] (kv_heads·dk), got %v/%v", kvDM, k.Shape(), v.Shape())
	}
	if sq > sk {
		return nil, fmt.Errorf("cpu: mha query len %d exceeds key len %d", sq, sk)
	}
	causal := pa.Causal
	scale := pa.Scale / math.Sqrt(float64(dk))
	off := sk - sq
	var slopes []float64
	if pa.ALiBi {
		slopes = backend.ALiBiSlopes(heads)
	}
	window := pa.Window

	qr, kr, vr := f64at(q.Contiguous()), f64at(k.Contiguous()), f64at(v.Contiguous())
	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{sq, dm})
	ow := f64set(out)

	parallelWork(heads*sq, sk*dk, func(lo, hi int) {
		row := make([]float64, sk)
		for t := lo; t < hi; t++ {
			h, i := t/sq, t%sq
			qOff := h * dk
			kvOff := (h / rep) * dk
			jmax := sk
			if causal {
				jmax = off + i + 1
			}
			jmin := 0
			if window > 0 {
				if lo := off + i - window + 1; lo > 0 {
					jmin = lo
				}
			}
			qBase := i*dm + qOff
			m := math.Inf(-1)
			for j := jmin; j < jmax; j++ {
				kBase := j*kvDM + kvOff
				var s float64
				for d := 0; d < dk; d++ {
					s += qr(qBase+d) * kr(kBase+d)
				}
				s *= scale
				if slopes != nil {
					s += slopes[h] * float64(j-(off+i))
				}
				row[j] = s
				if s > m {
					m = s
				}
			}
			var sum float64
			for j := jmin; j < jmax; j++ {
				row[j] = math.Exp(row[j] - m)
				sum += row[j]
			}
			for d := 0; d < dk; d++ {
				var o float64
				for j := jmin; j < jmax; j++ {
					o += (row[j] / sum) * vr(j*kvDM+kvOff+d)
				}
				ow(qBase+d, o)
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

// mhaBackwardKernelCPU: see the package comment above. Training has sq==sk.
func mhaBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("cpu: mha-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, g := in[0], in[1], in[2], in[3]
	seq, dm := q.Shape()[0], q.Shape()[1]
	if k.Shape()[0] != seq {
		return nil, fmt.Errorf("cpu: mha-backward needs equal Q/K length (got %d/%d); KV-cache is inference-only", seq, k.Shape()[0])
	}
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: mha-backward dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: mha-backward heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	kvDM := kvHeads * dk
	causal := pa.Causal
	scale := pa.Scale / math.Sqrt(float64(dk))
	var slopes []float64
	if pa.ALiBi {
		slopes = backend.ALiBiSlopes(heads)
	}
	window := pa.Window

	qr, kr, vr, gr := f64at(q.Contiguous()), f64at(k.Contiguous()), f64at(v.Contiguous()), f64at(g.Contiguous())
	dQ := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	dK := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())
	dV := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())
	dqw, dkw, dvw := f64set(dQ), f64set(dK), f64set(dV)

	// One task per KV group: its rep query heads touch exactly the group's
	// K/V columns and their own Q columns — fully disjoint output regions.
	parallelWork(kvHeads, rep*seq*seq*dk, func(glo, ghi int) {
		a := make([]float64, seq)
		dA := make([]float64, seq)
		dkAcc := make([]float64, seq*dk) // f64 accumulators for this group's dK/dV
		dvAcc := make([]float64, seq*dk)
		dqRow := make([]float64, dk)
		for kv := glo; kv < ghi; kv++ {
			kvOff := kv * dk
			for x := range dkAcc {
				dkAcc[x] = 0
				dvAcc[x] = 0
			}
			for hr := 0; hr < rep; hr++ {
				h := kv*rep + hr
				qOff := h * dk
				for i := 0; i < seq; i++ {
					jmax := seq
					if causal {
						jmax = i + 1
					}
					jmin := 0
					if window > 0 {
						if lo := i - window + 1; lo > 0 {
							jmin = lo
						}
					}
					qBase := i*dm + qOff
					m := math.Inf(-1)
					for j := jmin; j < jmax; j++ {
						kBase := j*kvDM + kvOff
						var s float64
						for d := 0; d < dk; d++ {
							s += qr(qBase+d) * kr(kBase+d)
						}
						s *= scale
						if slopes != nil {
							s += slopes[h] * float64(j-i)
						}
						a[j] = s
						if s > m {
							m = s
						}
					}
					var sum float64
					for j := jmin; j < jmax; j++ {
						a[j] = math.Exp(a[j] - m)
						sum += a[j]
					}
					for j := jmin; j < jmax; j++ {
						a[j] /= sum
					}
					var dot float64
					for j := jmin; j < jmax; j++ {
						kvBase := j*kvDM + kvOff
						accBase := j * dk
						var dav float64
						for d := 0; d < dk; d++ {
							gid := gr(qBase + d)
							dvAcc[accBase+d] += a[j] * gid
							dav += gid * vr(kvBase+d)
						}
						dA[j] = dav
						dot += dav * a[j]
					}
					for d := range dqRow {
						dqRow[d] = 0
					}
					for j := jmin; j < jmax; j++ {
						dS := scale * a[j] * (dA[j] - dot)
						kvBase := j*kvDM + kvOff
						accBase := j * dk
						for d := 0; d < dk; d++ {
							dqRow[d] += dS * kr(kvBase+d)
							dkAcc[accBase+d] += dS * qr(qBase+d)
						}
					}
					for d := 0; d < dk; d++ {
						dqw(qBase+d, dqRow[d])
					}
				}
			}
			for j := 0; j < seq; j++ {
				for d := 0; d < dk; d++ {
					dkw(j*kvDM+kvOff+d, dkAcc[j*dk+d])
					dvw(j*kvDM+kvOff+d, dvAcc[j*dk+d])
				}
			}
		}
	})
	return []*tensor.Tensor{dQ, dK, dV}, nil
}

func init() {
	std.add(backend.OpMHA, tensor.F32, mhaKernelCPU)
	std.add(backend.OpMHA, tensor.F64, mhaKernelCPU)
	std.add(backend.OpMHABackward, tensor.F32, mhaBackwardKernelCPU)
	std.add(backend.OpMHABackward, tensor.F64, mhaBackwardKernelCPU)
}
