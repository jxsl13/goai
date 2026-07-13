package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// (IA)³ rescale VJP (Liu et al. 2022, §R122). For y[...,j] = x[...,j]·l[j] with
// l[d] broadcast over the leading positions:
//
//	dx[...,j] = g[...,j]·l[j]                       // per-element scale
//	dl[j]     = Σ_rows g[row,j]·x[row,j]            // reduce over broadcast axis (§V10)
func init() {
	RegisterVJP(backend.OpIA3, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, l := in[0], in[1]
		d := x.Shape()[x.Ndim()-1]
		rows := x.Numel() / d

		dx := tensor.New(x.Dtype(), x.Shape())
		dl := tensor.New(l.Dtype(), l.Shape())
		acc := make([]float64, d) // f64 accumulation for dl
		for r := range rows {
			for j := range d {
				idx := tensor.Unravel(r*d+j, x.Shape())
				gv := g.AtF64(idx...)
				dx.SetF64(gv*l.AtF64(j), idx...)
				acc[j] += gv * x.AtF64(idx...)
			}
		}
		for j := range d {
			dl.SetF64(acc[j], j)
		}
		return []*tensor.Tensor{dx, dl}, nil
	})
}
