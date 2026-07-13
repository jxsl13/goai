package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// PPO clipped-surrogate VJP (Schulman et al. 2017, arXiv:1707.06347 Eq. 7). With
// rᵢ = exp(logπθᵢ − logπ_oldᵢ), surr1 = rᵢ·Âᵢ, surr2 = clip(rᵢ,1−ε,1+ε)·Âᵢ, and
// loss = −mean min(surr1, surr2), the gradient into the policy log-probs matches
// PyTorch autodiff of −min(surr1, surr2):
//
//	∂obj/∂logπθᵢ = Âᵢ·rᵢ        if surr1 selected, or surr2 selected & rᵢ ∈ (1−ε,1+ε)
//	             = 0            if surr2 selected & rᵢ clamped (flat clip → no update)
//	∂loss/∂logπθᵢ = g·(−1/N)·∂obj/∂logπθᵢ
//
// logπ_old and Â are rollout-time constants → nil gradients (§R49).
func init() {
	RegisterVJP(backend.OpPPOClip, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		lpNew, lpOld, adv := in[0], in[1], in[2]
		pX, _ := attrs.(backend.PPOClipAttrs)
		eps := pX.WithDefaults().Epsilon
		b := lpNew.Shape()[0]
		gv := g.AtF64()
		scale := -gv / float64(b)

		gNew := tensor.New(lpNew.Dtype(), lpNew.Shape())
		for i := range b {
			r := math.Exp(lpNew.AtF64(i) - lpOld.AtF64(i))
			a := adv.AtF64(i)
			surr1 := r * a
			surr2 := math.Max(1-eps, math.Min(1+eps, r)) * a

			var dobj float64
			if surr1 <= surr2 {
				dobj = a * r // unclipped branch (dr/dlogπ = r)
			} else if r > 1-eps && r < 1+eps {
				dobj = a * r // clipped branch but ratio inside the trust region
			} // else: clamped flat → 0
			gNew.SetF64(scale*dobj, i)
		}
		return []*tensor.Tensor{gNew, nil, nil}, nil // logπ_old, advantage frozen
	})
}
