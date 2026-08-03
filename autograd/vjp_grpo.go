package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GRPO VJP (Shao et al. 2024, DeepSeekMath, arXiv:2402.03300 Eqs. 3-4). The loss
// is the PPO clipped surrogate plus a k3 KL-to-reference penalty:
//
//	loss = −mean_i( surrᵢ − β·klᵢ ),  klᵢ = exp(Δᵢ) − Δᵢ − 1,  Δᵢ = logπ_refᵢ − logπθᵢ
//
// The surrogate gradient is exactly PPO's; the k3 term adds ∂klᵢ/∂logπθᵢ =
// 1 − exp(Δᵢ) (since ∂Δᵢ/∂logπθᵢ = −1), so:
//
//	∂loss/∂logπθᵢ = g·( −(1/N)·∂surrᵢ/∂logπθᵢ + (β/N)·(1 − exp(Δᵢ)) )
//
// logπ_old, logπ_ref and Â are rollout-time constants → nil gradients.
func init() {
	RegisterVJP(backend.OpGRPO, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		lpNew, lpOld, lpRef, adv := in[0], in[1], in[2], in[3]
		pX, _ := attrs.(backend.GRPOAttrs)
		pX = pX.WithDefaults()
		eps := pX.Epsilon
		beta := pX.Beta
		b := lpNew.Shape()[0]
		gv := g.AtF64()
		invN := gv / float64(b)

		gNew := tensor.New(lpNew.Dtype(), lpNew.Shape())
		lo, hi := 1-eps, 1+eps
		elemVJP(b, []*tensor.Tensor{lpNew, lpOld, lpRef, adv}, []*tensor.Tensor{gNew},
			func(in, out [][]float64, n int) {
				lnv, lov, lrv, av, o := in[0], in[1], in[2], in[3], out[0]
				for i := range n {
					r := math.Exp(lnv[i] - lov[i])
					a := av[i]
					// The clip as a comparison chain rather than math.Max(lo, math.Min(hi, r)):
					// `c <= lo` and not `<` is what reproduces math.Max on a negative zero, and
					// NaN falls through both bounds as math.Min and math.Max also leave it.
					c := r
					if c > hi {
						c = hi
					}
					if c <= lo {
						c = lo
					}
					dsurr := 0.0
					if r*a <= c*a {
						dsurr = a * r // unclipped branch (dr/dlogπ = r)
					} else if r > lo && r < hi {
						dsurr = a * r // clipped branch but ratio inside the trust region
					} // else: clamped flat → 0
					dkl := 1 - math.Exp(lrv[i]-lnv[i]) // ∂kl/∂logπθ = 1 − exp(Δ)
					o[i] = invN * (-dsurr + beta*dkl)
				}
			})
		return []*tensor.Tensor{gNew, nil, nil, nil}, nil // old, ref, advantage frozen
	})
}
