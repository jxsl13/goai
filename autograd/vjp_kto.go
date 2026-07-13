package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// KTO loss VJP (Ethayarajh et al. 2024). With r = β·(logπθ − logπref), detached
// reference point z_ref, and per-example loss λ_D·σ(z_ref−r) (desirable) or
// λ_U·σ(r−z_ref) (undesirable), the gradient into the policy log-probs is
//
//	desirable:   ∂L/∂logπθ = g·(1/N)·(−λ_D·β·s(1−s)),  s = σ(z_ref−r)
//	undesirable: ∂L/∂logπθ = g·(1/N)·( λ_U·β·s(1−s)),  s = σ(r−z_ref)
//
// The reference log-probs, labels, and z_ref are frozen/detached → nil (§R59).
func init() {
	RegisterVJP(backend.OpKTO, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		pl, rl, lab := in[0], in[1], in[2]
		pX, _ := attrs.(backend.KTOAttrs)
		pX = pX.WithDefaults()
		beta := pX.Beta
		zRef := pX.ZRef
		lambdaD := pX.LambdaD
		lambdaU := pX.LambdaU
		b := pl.Shape()[0]
		scale := g.AtF64() * beta / float64(b)

		gp := tensor.New(pl.Dtype(), pl.Shape())
		for i := range b {
			r := beta * (pl.AtF64(i) - rl.AtF64(i))
			if lab.AtF64(i) != 0 { // desirable: L = λ_D·σ(z_ref−r)
				s := sigmoidStable(zRef - r)
				gp.SetF64(-scale*lambdaD*s*(1-s), i)
			} else { // undesirable: L = λ_U·σ(r−z_ref)
				s := sigmoidStable(r - zRef)
				gp.SetF64(scale*lambdaU*s*(1-s), i)
			}
		}
		return []*tensor.Tensor{gp, nil, nil}, nil // reference, labels frozen
	})
}
