package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DPO loss VJP (Rafailov et al. 2023, arXiv:2305.18290, Eq. 7). With
// Δᵢ = β·((πcᵢ − rcᵢ) − (πlᵢ − rlᵢ)) and loss = mean_i −log σ(Δᵢ), the gradient
// w.r.t. the policy log-probs is
//
//	∂L/∂πcᵢ = g·(−β/N)·σ(−Δᵢ)
//	∂L/∂πlᵢ = g·(+β/N)·σ(−Δᵢ)
//
// i.e. weighted by σ of the REVERSED margin — largest when the policy wrongly
// prefers the rejected response (matches the paper's ∇θ expansion). The reference
// log-probs (rc, rl) are frozen → nil gradients.
func init() {
	RegisterVJP(backend.OpDPO, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		pc, pl, rc, rl := in[0], in[1], in[2], in[3]
		pX, _ := attrs.(backend.DPOAttrs)
		beta := pX.WithDefaults().Beta
		b := pc.Shape()[0]
		gv := g.AtF64() // scalar upstream gradient
		scale := gv * beta / float64(b)

		gpc := tensor.New(pc.Dtype(), pc.Shape())
		gpl := tensor.New(pl.Dtype(), pl.Shape())
		elemVJP(b, []*tensor.Tensor{pc, rc, pl, rl}, []*tensor.Tensor{gpc, gpl},
			func(in, out [][]float64, n int) {
				pcv, rcv, plv, rlv, a, c := in[0], in[1], in[2], in[3], out[0], out[1]
				//perfscan:ignore PS4003 DPO loss VJP over batch-length log-probs, tiny trip, once per train step
				for i := range n {
					delta := beta * ((pcv[i] - rcv[i]) - (plv[i] - rlv[i]))
					w := sigmoidStable(-delta) // σ(−Δᵢ)
					a[i] = -scale * w
					c[i] = scale * w
				}
			})
		return []*tensor.Tensor{gpc, gpl, nil, nil}, nil // reference frozen
	})
}

// sigmoidStable computes 1/(1+e^{-x}) without overflow for large |x|.
func sigmoidStable(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}
