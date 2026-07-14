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
		n := x.Numel()
		switch x.Dtype() {
		case tensor.F64:
			if g.Dtype() == tensor.F64 {
				xs := x.Contiguous().Storage().F64()
				gs := g.Contiguous().Storage().F64()
				ds := dx.Storage().F64()
				for i := 0; i < n; i++ {
					s := 1 / (1 + math.Exp(-xs[i]))
					ds[i] = gs[i] * s
				}
				return []*tensor.Tensor{dx}, nil
			}
		case tensor.F32:
			if g.Dtype() == tensor.F32 {
				xs := x.Contiguous().Storage().F32()
				gs := g.Contiguous().Storage().F32()
				ds := dx.Storage().F32()
				for i := 0; i < n; i++ {
					s := 1 / (1 + math.Exp(-float64(xs[i])))
					ds[i] = float32(float64(gs[i]) * s)
				}
				return []*tensor.Tensor{dx}, nil
			}
		}
		// generic fallback for exotic dtypes / mixed inputs
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

		switch x.Dtype() {
		case tensor.F64:
			if w.Dtype() == tensor.F64 && g.Dtype() == tensor.F64 && (!hasBias || in[2].Dtype() == tensor.F64) {
				xs := x.Contiguous().Storage().F64()
				ws := w.Contiguous().Storage().F64()
				gs := g.Contiguous().Storage().F64()
				dxs := dx.Storage().F64()
				dws := dw.Storage().F64()
				var dbs []float64
				if hasBias {
					dbs = db.Storage().F64()
				}
				for t := 0; t < L; t++ {
					for c := 0; c < D; c++ {
						gv := gs[t*D+c]
						if hasBias {
							dbs[c] += gv
						}
						for k := 0; k < K; k++ {
							j := t - (K - 1) + k
							if j >= 0 {
								dws[c*K+k] += gv * xs[j*D+c]
								dxs[j*D+c] += gv * ws[c*K+k]
							}
						}
					}
				}
				grads := []*tensor.Tensor{dx, dw}
				if hasBias {
					grads = append(grads, db)
				}
				return grads, nil
			}
		case tensor.F32:
			if w.Dtype() == tensor.F32 && g.Dtype() == tensor.F32 && (!hasBias || in[2].Dtype() == tensor.F32) {
				xs := x.Contiguous().Storage().F32()
				ws := w.Contiguous().Storage().F32()
				gs := g.Contiguous().Storage().F32()
				dxs := dx.Storage().F32()
				dws := dw.Storage().F32()
				var dbs []float32
				if hasBias {
					dbs = db.Storage().F32()
				}
				for t := 0; t < L; t++ {
					for c := 0; c < D; c++ {
						gv := float64(gs[t*D+c])
						if hasBias {
							dbs[c] = float32(float64(dbs[c]) + gv)
						}
						for k := 0; k < K; k++ {
							j := t - (K - 1) + k
							if j >= 0 {
								dws[c*K+k] = float32(float64(dws[c*K+k]) + gv*float64(xs[j*D+c]))
								dxs[j*D+c] = float32(float64(dxs[j*D+c]) + gv*float64(ws[c*K+k]))
							}
						}
					}
				}
				grads := []*tensor.Tensor{dx, dw}
				if hasBias {
					grads = append(grads, db)
				}
				return grads, nil
			}
		}

		// generic fallback for exotic dtypes / mixed inputs
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
