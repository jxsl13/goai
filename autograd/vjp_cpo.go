package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// CPO VJP — reference-free preference + NLL/BC regularizer over the summed
// chosen/rejected log-probs (Xu et al. 2024, §R121). Scalar loss; gradients flow
// into both policy log-prob vectors.
//
// loss = mean_i( softplus(−z_i) + λ·(−lw_i) ), z_i = β·(lw_i − ll_i). With
// σ(−z) = 1/(1+eᶻ) and ∂softplus(−z)/∂z = −σ(−z):
//
//	∂loss/∂lw = ( −β·σ(−z) − λ ) / N
//	∂loss/∂ll = (  β·σ(−z)     ) / N
func init() {
	RegisterVJP(backend.OpCPO, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		w, l := in[0], in[1]
		pX, _ := attrs.(backend.CPOAttrs)
		pX = pX.WithDefaults()
		beta, alpha := pX.Beta, pX.Alpha
		b := w.Shape()[0]
		scale := g.AtF64() / float64(b)

		dw := tensor.New(w.Dtype(), w.Shape())
		dl := tensor.New(l.Dtype(), l.Shape())
		elemVJP(b, []*tensor.Tensor{w, l}, []*tensor.Tensor{dw, dl},
			func(in, out [][]float64, n int) {
				wv, lv, dwv, dlv := in[0], in[1], out[0], out[1]
				for i := range n {
					z := beta * (wv[i] - lv[i])
					sNeg := 1 / (1 + math.Exp(z)) // σ(−z)
					dwv[i] = scale * (-beta*sNeg - alpha)
					dlv[i] = scale * (beta * sNeg)
				}
			})
		return []*tensor.Tensor{dw, dl}, nil
	})
}
