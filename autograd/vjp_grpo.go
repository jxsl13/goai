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
		for i := range b {
			r := math.Exp(lpNew.AtF64(i) - lpOld.AtF64(i))
			a := adv.AtF64(i)
			surr1 := r * a
			surr2 := math.Max(1-eps, math.Min(1+eps, r)) * a

			var dsurr float64
			if surr1 <= surr2 {
				dsurr = a * r // unclipped branch (dr/dlogπ = r)
			} else if r > 1-eps && r < 1+eps {
				dsurr = a * r // clipped branch but ratio inside the trust region
			} // else: clamped flat → 0

			d := lpRef.AtF64(i) - lpNew.AtF64(i)
			dkl := 1 - math.Exp(d) // ∂kl/∂logπθ = 1 − exp(Δ)
			gNew.SetF64(invN*(-dsurr+beta*dkl), i)
		}
		return []*tensor.Tensor{gNew, nil, nil, nil}, nil // old, ref, advantage frozen
	})
}
