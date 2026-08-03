package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// IPO loss VJP (Azar et al. 2023). With hᵢ = (πcᵢ−rcᵢ)−(πlᵢ−rlᵢ) and
// loss = mean_i (hᵢ − 1/(2β))², the gradient into the policy log-probs is
//
//	∂L/∂πcᵢ = g·(2/N)·(hᵢ − 1/(2β))
//	∂L/∂πlᵢ = −g·(2/N)·(hᵢ − 1/(2β))
//
// (h is linear in the policy log-probs, +1 for chosen, −1 for rejected). The
// reference log-probs are frozen → nil gradients (§R58).
func init() {
	RegisterVJP(backend.OpIPO, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		pc, pl, rc, rl := in[0], in[1], in[2], in[3]
		pX, _ := attrs.(backend.IPOAttrs)
		beta := pX.WithDefaults().Beta
		target := 1.0 / (2.0 * beta)
		b := pc.Shape()[0]
		scale := g.AtF64() * 2 / float64(b)

		gpc := tensor.New(pc.Dtype(), pc.Shape())
		gpl := tensor.New(pl.Dtype(), pl.Shape())
		elemVJP(b, []*tensor.Tensor{pc, rc, pl, rl}, []*tensor.Tensor{gpc, gpl},
			func(in, out [][]float64, n int) {
				pcv, rcv, plv, rlv, a, c := in[0], in[1], in[2], in[3], out[0], out[1]
				for i := range n {
					h := (pcv[i] - rcv[i]) - (plv[i] - rlv[i])
					d := scale * (h - target)
					a[i] = d
					c[i] = -d
				}
			})
		return []*tensor.Tensor{gpc, gpl, nil, nil}, nil // reference frozen
	})
}
