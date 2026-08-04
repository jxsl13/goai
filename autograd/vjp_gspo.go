package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GSPO VJP (§T549): with s_i = exp(mean_t Δ_t) the sequence ratio, every token t of
// sequence i shares the same gradient
//
//	∂loss/∂logπθ_t = −(g/G) · Â_i · s_i / |y_i|     (when the surrogate is live)
//
// and 0 when the min picks the SATURATED clip branch (s_i outside [1−ε,1+ε] and the
// clipped term is smaller) — the same branch convention as the GRPO/PPO VJPs.
// logπ_old and the advantages are rollout-time constants.
func init() {
	RegisterVJP(backend.OpGSPO, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		lpNew, lpOld, adv := in[0], in[1], in[2]
		pa, _ := attrs.(backend.GSPOAttrs)
		pa = pa.WithDefaults()
		eps := pa.Epsilon
		gv := g.AtF64()
		invG := gv / float64(len(pa.Lengths))

		gNew := tensor.New(lpNew.Dtype(), lpNew.Shape())
		// The per-token gradient is a single value `per` shared by every token of a
		// sequence, but the ratio pass read lpNew/lpOld through AtF64 and the write
		// pass stored it through gNew.SetF64 — interface dispatch over the whole token
		// axis. When the logp tensors are contiguous F64/F32, walk the storage slices
		// and fill each sequence span with the constant. Bit-identical: the Δ sum keeps
		// the same token order and `per` is computed with the same exp/min/max/branch.
		var lpNewF64, lpOldF64, gF64 []float64
		var lpNewF32, lpOldF32, gF32 []float32
		typedF64 := lpNew.Dtype() == tensor.F64 && lpOld.Dtype() == tensor.F64
		typedF32 := !typedF64 && lpNew.Dtype() == tensor.F32 && lpOld.Dtype() == tensor.F32
		if typedF64 {
			lpNewF64 = lpNew.Contiguous().Storage().F64()
			lpOldF64 = lpOld.Contiguous().Storage().F64()
			gF64 = gNew.Storage().F64()
		} else if typedF32 {
			lpNewF32 = lpNew.Contiguous().Storage().F32()
			lpOldF32 = lpOld.Contiguous().Storage().F32()
			gF32 = gNew.Storage().F32()
		}
		off := 0
		for i, l := range pa.Lengths {
			var d float64
			switch {
			case typedF64:
				//perfscan:ignore PS3010 already typed F64 flat-slice fast path (the optimization itself)
				for t := 0; t < l; t++ {
					d += lpNewF64[off+t] - lpOldF64[off+t]
				}
			case typedF32:
				//perfscan:ignore PS3010 already typed F32 flat-slice fast path
				for t := 0; t < l; t++ {
					d += float64(lpNewF32[off+t]) - float64(lpOldF32[off+t])
				}
			default:
				for t := range l {
					d += lpNew.AtF64(off+t) - lpOld.AtF64(off+t)
				}
			}
			s := math.Exp(d / float64(l))
			a := adv.AtF64(i)
			surr1 := s * a
			//perfscan:ignore PS3077,PS3082 per-sequence scalar min/max (O(#seq)), not in token loop | per-sequence scalar clip, negligible
			surr2 := math.Max(1-eps, math.Min(1+eps, s)) * a

			var dsurr float64
			if surr1 <= surr2 {
				dsurr = a * s // unclipped branch; ds/dlogπ_t = s/|y|
			} else if s > 1-eps && s < 1+eps {
				dsurr = a * s // clip is identity inside the trust region
			} // else: saturated clip → 0
			per := invG * -dsurr / float64(l)
			switch {
			case typedF64:
				for t := 0; t < l; t++ {
					gF64[off+t] = per
				}
			case typedF32:
				pf := float32(per)
				for t := 0; t < l; t++ {
					gF32[off+t] = pf
				}
			default:
				for t := range l {
					gNew.SetF64(per, off+t)
				}
			}
			off += l
		}
		return []*tensor.Tensor{gNew, nil, nil}, nil // old and advantage frozen
	})
}
