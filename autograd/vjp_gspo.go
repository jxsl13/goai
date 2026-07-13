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
		off := 0
		for i, l := range pa.Lengths {
			var d float64
			for t := range l {
				d += lpNew.AtF64(off+t) - lpOld.AtF64(off+t)
			}
			s := math.Exp(d / float64(l))
			a := adv.AtF64(i)
			surr1 := s * a
			surr2 := math.Max(1-eps, math.Min(1+eps, s)) * a

			var dsurr float64
			if surr1 <= surr2 {
				dsurr = a * s // unclipped branch; ds/dlogπ_t = s/|y|
			} else if s > 1-eps && s < 1+eps {
				dsurr = a * s // clip is identity inside the trust region
			} // else: saturated clip → 0
			per := invG * -dsurr / float64(l)
			for t := range l {
				gNew.SetF64(per, off+t)
			}
			off += l
		}
		return []*tensor.Tensor{gNew, nil, nil}, nil // old and advantage frozen
	})
}
