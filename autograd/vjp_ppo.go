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
		lo, hi := 1-eps, 1+eps
		// step is one element's contribution, written once and shared by both arms below so the
		// typed path cannot drift from the accessor path.
		step := func(lpn, lpo, a float64) float64 {
			r := math.Exp(lpn - lpo)
			// THE CLAMP IS A COMPARISON CHAIN, not math.Max(lo, math.Min(hi, r)): those are two
			// calls per element carrying the full NaN and signed-zero contract. `c <= lo` rather
			// than `<` is what reproduces math.Max on a negative zero, and NaN compares false
			// against both bounds so it falls through unchanged, which is what math.Min and
			// math.Max also return when either operand is NaN.
			c := r
			if c > hi {
				c = hi
			}
			if c <= lo {
				c = lo
			}
			if r*a <= c*a { // unclipped branch (dr/dlogπ = r)
				return a * r
			}
			if r > lo && r < hi { // clipped branch but ratio inside the trust region
				return a * r
			}
			return 0 // clamped flat
		}
		// TYPED STORAGE WHEN EVERY OPERAND IS ALREADY F64. The accessor arm below pays four
		// dispatches per element — three AtF64 reads and a SetF64 — each walking the shape to a
		// flat offset and switching on the storage type. The dtype comes from the input, so the
		// accessor arm has to stay for anything else.
		if lpNew.Dtype() == tensor.F64 && lpOld.Dtype() == tensor.F64 && adv.Dtype() == tensor.F64 {
			ln := lpNew.Contiguous().Storage().F64()
			lo64 := lpOld.Contiguous().Storage().F64()
			av := adv.Contiguous().Storage().F64()
			out := gNew.Storage().F64()
			for i := range b {
				out[i] = scale * step(ln[i], lo64[i], av[i])
			}
			return []*tensor.Tensor{gNew, nil, nil}, nil
		}
		for i := range b {
			gNew.SetF64(scale*step(lpNew.AtF64(i), lpOld.AtF64(i), adv.AtF64(i)), i)
		}
		return []*tensor.Tensor{gNew, nil, nil}, nil // logπ_old, advantage frozen
	})
}
