package autograd

import (

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// CV VJPs (§T24b, closes §B24). Direct-loop implementations mirroring the
// forward geometry, accumulating in f64 (§V10):
//
//	conv2d:  gx[n,c,iy,ix] += g·w[f,c,ky,kx];  gw += g·x;  gb[f] = Σ g[n,f,·,·]
//	maxpool: g routes to the FIRST window element attaining the max (§B16 rule)
//	avgpool: g/k² spreads uniformly over the window

func init() {
	RegisterVJP(backend.OpConv2D, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, w := in[0], in[1]
		n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
		f, kh, kw := w.Shape()[0], w.Shape()[2], w.Shape()[3]
		s := attrs.Int("stride", 1)
		p := attrs.Int("pad", 0)
		ho, wo := g.Shape()[2], g.Shape()[3]

		gx := tensor.New(x.Dtype(), x.Shape())
		gw := tensor.New(w.Dtype(), w.Shape())
		var gb *tensor.Tensor
		if len(in) == 3 {
			gb = tensor.New(in[2].Dtype(), in[2].Shape())
		}
		for ni := range n {
			for fi := range f {
				var bsum float64
				for oy := range ho {
					for ox := range wo {
						gv := g.AtF64(ni, fi, oy, ox)
						bsum += gv
						for ci := range c {
							for ky := range kh {
								iy := oy*s + ky - p
								if iy < 0 || iy >= h {
									continue
								}
								for kx := range kw {
									ix := ox*s + kx - p
									if ix < 0 || ix >= wd {
										continue
									}
									gx.SetF64(gx.AtF64(ni, ci, iy, ix)+gv*w.AtF64(fi, ci, ky, kx), ni, ci, iy, ix)
									gw.SetF64(gw.AtF64(fi, ci, ky, kx)+gv*x.AtF64(ni, ci, iy, ix), fi, ci, ky, kx)
								}
							}
						}
					}
				}
				if gb != nil {
					gb.SetF64(gb.AtF64(fi)+bsum, fi)
				}
			}
		}
		grads := []*tensor.Tensor{gx, gw}
		if len(in) == 3 {
			grads = append(grads, gb)
		}
		return grads, nil
	})

	RegisterVJP(backend.OpMaxPool2D, func(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		n, c := x.Shape()[0], x.Shape()[1]
		k := attrs.Int("kernel", 0)
		s := attrs.Int("stride", k)
		ho, wo := y.Shape()[2], y.Shape()[3]
		gx := tensor.New(x.Dtype(), x.Shape())
		for ni := range n {
			for ci := range c {
				for oy := range ho {
					for ox := range wo {
						m := y.AtF64(ni, ci, oy, ox)
						routed := false
						for ky := 0; ky < k && !routed; ky++ {
							for kx := 0; kx < k && !routed; kx++ {
								iy, ix := oy*s+ky, ox*s+kx
								if x.AtF64(ni, ci, iy, ix) == m {
									gx.SetF64(gx.AtF64(ni, ci, iy, ix)+g.AtF64(ni, ci, oy, ox), ni, ci, iy, ix)
									routed = true
								}
							}
						}
						if !routed { // NaN window: max is NaN, == fails; route to first
							gx.SetF64(gx.AtF64(ni, ci, oy*s, ox*s)+g.AtF64(ni, ci, oy, ox), ni, ci, oy*s, ox*s)
						}
					}
				}
			}
		}
		return []*tensor.Tensor{gx}, nil
	})

	RegisterVJP(backend.OpAvgPool2D, func(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		n, c := x.Shape()[0], x.Shape()[1]
		k := attrs.Int("kernel", 0)
		s := attrs.Int("stride", k)
		ho, wo := y.Shape()[2], y.Shape()[3]
		inv := 1 / float64(k*k)
		gx := tensor.New(x.Dtype(), x.Shape())
		for ni := range n {
			for ci := range c {
				for oy := range ho {
					for ox := range wo {
						gv := g.AtF64(ni, ci, oy, ox) * inv
						for ky := range k {
							for kx := range k {
								iy, ix := oy*s+ky, ox*s+kx
								gx.SetF64(gx.AtF64(ni, ci, iy, ix)+gv, ni, ci, iy, ix)
							}
						}
					}
				}
			}
		}
		return []*tensor.Tensor{gx}, nil
	})
}
