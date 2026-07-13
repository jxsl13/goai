package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaBackwardKernel is the fused scaled-dot-product-attention backward (§T86, the
// gradient of §T32's OpMHA). Inputs are (Q, K, V, dO) each [seq, dmodel] (dO is
// the upstream gradient of the attention output); it returns (dQ, dK, dV). Making
// the backward a dispatched op — instead of hand-rolled tensor loops inside the
// VJP — lets it run on whichever backend is active (GPU when training on Metal,
// §T86), exactly as the matmul backward already does.
//
// For each head, recompute the softmax weights A, then (standard SDPA gradient):
//
//	dV = Aᵀ·dO ;  dA = dO·Vᵀ ;  dS = A⊙(dA − rowsum(dA⊙A)) ;
//	dQ = scale·dS·K ;  dK = scale·dSᵀ·Q
//
// Causal masking drops j>i (A there is 0), so no gradient flows to masked pairs.
// ALiBi (§R60) and sliding-window (§R62) match the forward. f64 accumulation
// (§V10). Training always has sq==sk (KV-cache decode is inference-only).
func mhaBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("ref: mha-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, g := in[0], in[1], in[2], in[3]
	seq, dm := q.Shape()[0], q.Shape()[1]
	if k.Shape()[0] != seq {
		return nil, fmt.Errorf("ref: mha-backward needs equal Q/K length (got %d/%d); KV-cache is inference-only", seq, k.Shape()[0])
	}
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("ref: mha-backward dmodel %d not divisible by heads %d", dm, heads)
	}
	causal := pa.Causal
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("ref: mha-backward heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	// same YaRN attention-temperature scaling as the forward (§R66), folded into
	// the 1/√dk scale so the backward matches the scaled softmax.
	scale := pa.Scale / math.Sqrt(float64(dk))
	var slopes []float64
	if pa.ALiBi {
		slopes = backend.ALiBiSlopes(heads)
	}
	window := pa.Window

	dQ := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	dK := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())
	dV := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())

	a := make([]float64, seq)  // softmax weights for the current (head,row)
	dA := make([]float64, seq) // grad w.r.t. those weights
	for h := range heads {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := range seq {
			jmax := seq
			if causal {
				jmax = i + 1
			}
			jmin := 0
			if window > 0 {
				if lo := i - window + 1; lo > 0 { // sq==sk → abs pos = i
					jmin = lo
				}
			}
			// recompute A[i,:] (same as forward)
			m := math.Inf(-1)
			for j := jmin; j < jmax; j++ {
				var s float64
				for d := range dk {
					s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
				}
				s *= scale
				if slopes != nil {
					s += slopes[h] * float64(j-i) // ALiBi bias (sq==sk → abs pos = i)
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
			// dV[j] += A[i,j]·dO[i] ; dA[i,j] = dO[i]·V[j]
			var dot float64
			for j := jmin; j < jmax; j++ {
				var dav float64
				for d := range dk {
					gid := g.AtF64(i, qOff+d)
					dV.SetF64(dV.AtF64(j, kvOff+d)+a[j]*gid, j, kvOff+d)
					dav += gid * v.AtF64(j, kvOff+d)
				}
				dA[j] = dav
				dot += dav * a[j]
			}
			// dS = A⊙(dA − dot); dQ/dK via bilinear form with scale
			for j := jmin; j < jmax; j++ {
				dS := scale * a[j] * (dA[j] - dot)
				for d := range dk {
					dQ.SetF64(dQ.AtF64(i, qOff+d)+dS*k.AtF64(j, kvOff+d), i, qOff+d)
					dK.SetF64(dK.AtF64(j, kvOff+d)+dS*q.AtF64(i, qOff+d), j, kvOff+d)
				}
			}
		}
	}
	return []*tensor.Tensor{dQ, dK, dV}, nil
}

func init() {
	std.add(backend.OpMHABackward, tensor.F32, mhaBackwardKernel)
	std.add(backend.OpMHABackward, tensor.F64, mhaBackwardKernel)
}
