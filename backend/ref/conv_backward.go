package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// conv2dBackwardKernel is the 2-D convolution backward (§T101, the gradient of
// §T88's OpConv2D). Inputs are (X[N,C,H,W], W[F,C,KH,KW], dO[N,F,ho,wo]); it always
// returns (dX, dW, dBias[F]) — the conv2d VJP drops dBias when the forward had no
// bias. Making the backward a dispatched op (like the matmul and attention
// backwards) lets it run on whichever backend is active. f64 accumulation (§V10):
//
//	dX[n,c,iy,ix] += dO[n,f,oy,ox]·W[f,c,ky,kx]
//	dW[f,c,ky,kx] += dO[n,f,oy,ox]·X[n,c,iy,ix]
//	dBias[f]       = Σ dO[n,f,·,·]
func conv2dBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: conv2d-backward wants (X,W,dO), got %d inputs", len(in))
	}
	x, w, g := in[0], in[1], in[2]
	if x.Ndim() != 4 || w.Ndim() != 4 || g.Ndim() != 4 {
		return nil, fmt.Errorf("ref: conv2d-backward needs rank-4 X,W,dO")
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("ref: conv2d-backward channel mismatch x C=%d vs w C=%d", c, wc)
	}
	pa, _ := attrs.(backend.ConvAttrs)
	pa = pa.WithDefaults()
	s, p := pa.Stride, pa.Pad
	ho, wo := g.Shape()[2], g.Shape()[3]

	dX := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	dW := tensor.NewOn(ctx.Device(), w.Dtype(), w.Shape())
	dB := tensor.NewOn(ctx.Device(), w.Dtype(), tensor.Shape{f})

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
								dX.SetF64(dX.AtF64(ni, ci, iy, ix)+gv*w.AtF64(fi, ci, ky, kx), ni, ci, iy, ix)
								dW.SetF64(dW.AtF64(fi, ci, ky, kx)+gv*x.AtF64(ni, ci, iy, ix), fi, ci, ky, kx)
							}
						}
					}
				}
			}
			dB.SetF64(dB.AtF64(fi)+bsum, fi)
		}
	}
	return []*tensor.Tensor{dX, dW, dB}, nil
}

func init() {
	std.add(backend.OpConv2DBackward, tensor.F32, conv2dBackwardKernel)
	std.add(backend.OpConv2DBackward, tensor.F64, conv2dBackwardKernel)
}
