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
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
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

	// Devirtualised typed core (§T646 follow-up): inputs exposed once as flat
	// []float64 (exact widening), gradients accumulated through a generic
	// refFloat core so the F32 instantiation reproduces the generic path's
	// per-update widen/add/narrow sequence exactly — bit-identical.
	if x.Dtype() == w.Dtype() && x.Dtype() == g.Dtype() {
		if xs, ok := f64Data(x); ok {
			ws, _ := f64Data(w)
			gs, _ := f64Data(g)
			switch x.Dtype() {
			case tensor.F64:
				conv2dBackwardCore(xs, ws, gs,
					dX.Storage().F64(), dW.Storage().F64(), dB.Storage().F64(),
					n, c, h, wd, f, kh, kw, ho, wo, s, p)
			case tensor.F32:
				conv2dBackwardCore(xs, ws, gs,
					dX.Storage().F32(), dW.Storage().F32(), dB.Storage().F32(),
					n, c, h, wd, f, kh, kw, ho, wo, s, p)
			}
			return []*tensor.Tensor{dX, dW, dB}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
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

// conv2dBackwardCore runs the conv2d backward over flat row-major slices. Reads
// come from exact-widened []float64; writes go to []T with every intermediate
// store narrowed for T=float32 (identical to the generic SetF64(AtF64+δ)
// chain). Loop structure and accumulation order match the generic loops exactly
// — bit-identical.
func conv2dBackwardCore[T refFloat](xs, ws, gs []float64, dxs, dws, dbs []T,
	n, c, h, wd, f, kh, kw, ho, wo, s, p int) {
	for ni := 0; ni < n; ni++ {
		for fi := 0; fi < f; fi++ {
			var bsum float64
			for oy := 0; oy < ho; oy++ {
				for ox := 0; ox < wo; ox++ {
					gv := gs[((ni*f+fi)*ho+oy)*wo+ox]
					bsum += gv
					for ci := 0; ci < c; ci++ {
						xplane := xs[(ni*c+ci)*h*wd : (ni*c+ci+1)*h*wd]
						dxplane := dxs[(ni*c+ci)*h*wd : (ni*c+ci+1)*h*wd]
						for ky := 0; ky < kh; ky++ {
							iy := oy*s + ky - p
							if iy < 0 || iy >= h {
								continue
							}
							wrow := ws[((fi*c+ci)*kh+ky)*kw : ((fi*c+ci)*kh+ky)*kw+kw]
							dwrow := dws[((fi*c+ci)*kh+ky)*kw : ((fi*c+ci)*kh+ky)*kw+kw]
							for kx := 0; kx < kw; kx++ {
								ix := ox*s + kx - p
								if ix < 0 || ix >= wd {
									continue
								}
								dxplane[iy*wd+ix] = T(float64(dxplane[iy*wd+ix]) + gv*wrow[kx])
								dwrow[kx] = T(float64(dwrow[kx]) + gv*xplane[iy*wd+ix])
							}
						}
					}
				}
			}
			dbs[fi] = T(float64(dbs[fi]) + bsum)
		}
	}
}

func init() {
	std.add(backend.OpConv2DBackward, tensor.F32, conv2dBackwardKernel)
	std.add(backend.OpConv2DBackward, tensor.F64, conv2dBackwardKernel)
}
