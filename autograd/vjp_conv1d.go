package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func init() {
	// softplus'(x) = σ(x) = 1/(1+e⁻ˣ)
	RegisterVJP(backend.OpSoftplus, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		dx := tensor.New(x.Dtype(), x.Shape())
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			s := 1 / (1 + math.Exp(-x.AtF64(idx...)))
			dx.SetF64(g.AtF64(idx...)*s, idx...)
		}
		return []*tensor.Tensor{dx}, nil
	})

	// Causal depthwise conv1d VJP (§R77). out[t,c]=Σ_k w[c,k]·x[t−(K−1)+k,c]+b[c]:
	//	db[c]   = Σ_t g[t,c]
	//	dw[c,k] = Σ_t g[t,c]·x[t−(K−1)+k, c]
	//	dx[j,c] = Σ_{(t,k): t−(K−1)+k=j} g[t,c]·w[c,k]
	RegisterVJP(backend.OpConv1D, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, w := in[0], in[1]
		hasBias := len(in) == 3
		L, D := x.Shape()[0], x.Shape()[1]
		K := w.Shape()[1]

		dx := tensor.New(x.Dtype(), x.Shape())
		dw := tensor.New(w.Dtype(), w.Shape())
		var db *tensor.Tensor
		if hasBias {
			db = tensor.New(in[2].Dtype(), in[2].Shape())
		}
		for t := range L {
			for c := range D {
				gv := g.AtF64(t, c)
				if hasBias {
					db.SetF64(db.AtF64(c)+gv, c)
				}
				for k := range K {
					j := t - (K - 1) + k
					if j >= 0 {
						dw.SetF64(dw.AtF64(c, k)+gv*x.AtF64(j, c), c, k)
						dx.SetF64(dx.AtF64(j, c)+gv*w.AtF64(c, k), j, c)
					}
				}
			}
		}
		grads := []*tensor.Tensor{dx, dw}
		if hasBias {
			grads = append(grads, db)
		}
		return grads, nil
	})
}
