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

	geo := mhaGeo{sq: sq, sk: sk, dm: dm, dk: dk, kvDM: kvDM, heads: heads, rep: rep,
		off: off, window: window, causal: causal, scale: scale, slopes: slopes}
	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{sq, dm})
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	// Concrete typed instantiations (§T602): the previous per-element closure
	// reads were 11% of the training profile; generics devirtualize them.
	if q.Dtype() == tensor.F64 {
		mhaFwd(qc.Storage().F64(), kc.Storage().F64(), vc.Storage().F64(), out.Storage().F64(), geo)
	} else {
		mhaFwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), out.Storage().F32(), geo)
	}
	return []*tensor.Tensor{out}, nil
}

// mhaGeo bundles the attention geometry and attributes shared by the typed cores.
type mhaGeo struct {
	sq, sk, dm, dk, kvDM, heads, rep, off, window int
	causal                                        bool
	scale                                         float64
	slopes                                        []float64
}

// bounds returns the [jmin, jmax) key range for query row i.
func (g mhaGeo) bounds(i int) (int, int) {
	jmax := g.sk
	if g.causal {
		jmax = g.off + i + 1
	}
	jmin := 0
	if g.window > 0 {
		if lo := g.off + i - g.window + 1; lo > 0 {
			jmin = lo
		}
	}
	return jmin, jmax
}

// mhaFwd is the typed attention forward over raw slices (f64 accumulation, §V10).
func mhaFwd[T float32 | float64](q, k, v, out []T, g mhaGeo) {
	parallelWork(g.heads*g.sq, g.sk*g.dk, func(lo, hi int) {
		row := make([]float64, g.sk)
		for t := lo; t < hi; t++ {
			h, i := t/g.sq, t%g.sq
			qOff := h * g.dk
			kvOff := (h / g.rep) * g.dk
			jmin, jmax := g.bounds(i)
			qBase := i*g.dm + qOff
			m := math.Inf(-1)
			for j := jmin; j < jmax; j++ {
				kBase := j*g.kvDM + kvOff
				var s float64
				for d := 0; d < g.dk; d++ {
					s += float64(q[qBase+d]) * float64(k[kBase+d])
				}
				s *= g.scale
				if g.slopes != nil {
					s += g.slopes[h] * float64(j-(g.off+i))
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
			for d := 0; d < g.dk; d++ {
				var o float64
				for j := jmin; j < jmax; j++ {
					o += (row[j] / sum) * float64(v[j*g.kvDM+kvOff+d])
				}
				out[qBase+d] = T(o)
			}
		}
	})
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

	dQ := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	dK := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())
	dV := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())
	geo := mhaGeo{sq: seq, sk: seq, dm: dm, dk: dk, kvDM: kvDM, heads: heads, rep: rep,
		off: 0, window: window, causal: causal, scale: scale, slopes: slopes}
	qc, kc, vc, gc := q.Contiguous(), k.Contiguous(), v.Contiguous(), g.Contiguous()
	if q.Dtype() == tensor.F64 {
		mhaBwd(qc.Storage().F64(), kc.Storage().F64(), vc.Storage().F64(), gc.Storage().F64(),
			dQ.Storage().F64(), dK.Storage().F64(), dV.Storage().F64(), geo)
	} else {
		mhaBwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), gc.Storage().F32(),
			dQ.Storage().F32(), dK.Storage().F32(), dV.Storage().F32(), geo)
	}
	return []*tensor.Tensor{dQ, dK, dV}, nil
}

// mhaBwd is the typed attention backward over raw slices (§T602); kvHeads
// parallel tasks with disjoint output regions, f64 scratch accumulation (§V10).
func mhaBwd[T float32 | float64](q, k, v, g, dQ, dK, dV []T, geo mhaGeo) {
	seq, dk, dm, kvDM, rep := geo.sq, geo.dk, geo.dm, geo.kvDM, geo.rep
	kvHeads := geo.heads / rep

	// One task per KV group: its rep query heads touch exactly the group's
	// K/V columns and their own Q columns — fully disjoint output regions.
	parallelWork(kvHeads, rep*seq*seq*dk, func(glo, ghi int) {
		causal, window, scale, slopes := geo.causal, geo.window, geo.scale, geo.slopes
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
			for hr := range rep {
				h := kv*rep + hr
				qOff := h * dk
				for i := range seq {
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
						for d := range dk {
							s += float64(q[qBase+d]) * float64(k[kBase+d])
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
						for d := range dk {
							gid := float64(g[qBase+d])
							dvAcc[accBase+d] += a[j] * gid
							dav += gid * float64(v[kvBase+d])
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
						for d := range dk {
							dqRow[d] += dS * float64(k[kvBase+d])
							dkAcc[accBase+d] += dS * float64(q[qBase+d])
						}
					}
					for d := range dk {
						dQ[qBase+d] = T(dqRow[d])
					}
				}
			}
			for j := range seq {
				for d := range dk {
					dK[j*kvDM+kvOff+d] = T(dkAcc[j*dk+d])
					dV[j*kvDM+kvOff+d] = T(dvAcc[j*dk+d])
				}
			}
		}
	})
}

func init() {
	std.add(backend.OpMHA, tensor.F32, mhaKernelCPU)
	std.add(backend.OpMHA, tensor.F64, mhaKernelCPU)
	std.add(backend.OpMHABackward, tensor.F32, mhaBackwardKernelCPU)
	std.add(backend.OpMHABackward, tensor.F64, mhaBackwardKernelCPU)
}
