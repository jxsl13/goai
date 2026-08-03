package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SimPO and ORPO VJPs — reference-free preference losses over length-averaged
// chosen/rejected log-probs (Meng et al. 2024 §R71; Hong et al. 2024 §R71). Both
// are scalar losses; gradients flow into the two policy log-prob vectors.
func init() {
	// SimPO: z = β·(avgW − avgL) − γ, loss = mean softplus(−z). With
	// σ(−z) = 1/(1+eᶻ): ∂loss/∂avgW = −β·σ(−z)/N, ∂loss/∂avgL = +β·σ(−z)/N.
	RegisterVJP(backend.OpSimPO, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		w, l := in[0], in[1]
		pX, _ := attrs.(backend.SimPOAttrs)
		pX = pX.WithDefaults()
		beta := pX.Beta
		gamma := pX.Gamma
		b := w.Shape()[0]
		scale := g.AtF64() / float64(b)

		dw := tensor.New(w.Dtype(), w.Shape())
		dl := tensor.New(l.Dtype(), l.Shape())
		elemVJP(b, []*tensor.Tensor{w, l}, []*tensor.Tensor{dw, dl},
			func(in, out [][]float64, n int) {
				wv, lv, dwv, dlv := in[0], in[1], out[0], out[1]
				for i := range n {
					z := beta*(wv[i]-lv[i]) - gamma
					sNeg := 1 / (1 + math.Exp(z)) // σ(−z)
					dwv[i] = scale * (-beta * sNeg)
					dlv[i] = scale * (beta * sNeg)
				}
			})
		return []*tensor.Tensor{dw, dl}, nil
	})

	// ORPO: loss = mean( −avgW + λ·softplus(−logOdds) ), logOdds = (avgW−avgL) −
	// (log(1−Pw) − log(1−Pl)), P = exp(avg). With s = σ(−logOdds) and
	// ∂logOdds/∂avgW = 1/(1−Pw), ∂logOdds/∂avgL = −1/(1−Pl):
	//	∂loss/∂avgW = −1 − λ·s/(1−Pw) ,  ∂loss/∂avgL = λ·s/(1−Pl)   (each /N)
	RegisterVJP(backend.OpORPO, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		w, l := in[0], in[1]
		pX, _ := attrs.(backend.ORPOAttrs)
		lambda := pX.WithDefaults().Lambda
		b := w.Shape()[0]
		scale := g.AtF64() / float64(b)

		dw := tensor.New(w.Dtype(), w.Shape())
		dl := tensor.New(l.Dtype(), l.Shape())
		elemVJP(b, []*tensor.Tensor{w, l}, []*tensor.Tensor{dw, dl},
			func(in, out [][]float64, n int) {
				wv, lv, dwv, dlv := in[0], in[1], out[0], out[1]
				for i := range n {
					aw, al := wv[i], lv[i]
					pw, pl := math.Exp(aw), math.Exp(al)
					logOdds := (aw - al) - (math.Log1p(-pw) - math.Log1p(-pl))
					s := 1 / (1 + math.Exp(logOdds)) // σ(−logOdds)
					dwv[i] = scale * (-1 - lambda*s/(1-pw))
					dlv[i] = scale * (lambda * s / (1 - pl))
				}
			})
		return []*tensor.Tensor{dw, dl}, nil
	})
}
