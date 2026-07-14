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
		// Typed fast paths (dtype-switch once → []T flat, no per-element Unravel/
		// AtF64/SetF64; acc stays f64, single stores round once, §base-perf/C25/§T642).
		xc, lc, gc := x.Contiguous(), l.Contiguous(), g.Contiguous()
		switch x.Dtype() {
		case tensor.F64:
			if lc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
				xs, ls, gs := xc.Storage().F64(), lc.Storage().F64(), gc.Storage().F64()
				dxs, dls := dx.Storage().F64(), dl.Storage().F64()
				for r := 0; r < rows; r++ {
					base := r * d
					for j := 0; j < d; j++ {
						gv := gs[base+j]
						dxs[base+j] = gv * ls[j]
						acc[j] += gv * xs[base+j]
					}
				}
				for j := 0; j < d; j++ {
					dls[j] = acc[j]
				}
				return []*tensor.Tensor{dx, dl}, nil
			}
		case tensor.F32:
			if lc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
				xs, ls, gs := xc.Storage().F32(), lc.Storage().F32(), gc.Storage().F32()
				dxs, dls := dx.Storage().F32(), dl.Storage().F32()
				for r := 0; r < rows; r++ {
					base := r * d
					for j := 0; j < d; j++ {
						gv := float64(gs[base+j])
						dxs[base+j] = float32(gv * float64(ls[j]))
						acc[j] += gv * float64(xs[base+j])
					}
				}
				for j := 0; j < d; j++ {
					dls[j] = float32(acc[j])
				}
				return []*tensor.Tensor{dx, dl}, nil
			}
		}
		for r := range rows { // generic fallback (exotic/mixed dtype)
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
