package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// CV reference kernels (§T24), NCHW layout, direct loops with f64 accumulation
// (§V10) — the truth for the later im2col/Winograd optimization task. Conv2D is
// cross-correlation (the DL convention, matching torch.nn.functional.conv2d):
// no kernel flip.

// conv2dKernel: x[N,C,H,W] ⋆ w[F,C,KH,KW] (+ bias[F]) → out[N,F,Ho,Wo] with
// Ho = (H+2p−KH)/s+1. Attrs: "stride" (default 1), "pad" (default 0).
func conv2dKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 && len(in) != 3 {
		return nil, fmt.Errorf("ref: conv2d wants (x, w[, bias]), got %d inputs", len(in))
	}
	x, w := in[0], in[1]
	if x.Ndim() != 4 || w.Ndim() != 4 {
		return nil, fmt.Errorf("ref: conv2d needs x[N,C,H,W] and w[F,C,KH,KW], got %v/%v", x.Shape(), w.Shape())
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("ref: conv2d channel mismatch x C=%d vs w C=%d", c, wc)
	}
	var bias *tensor.Tensor
	if len(in) == 3 {
		bias = in[2]
		if bias.Ndim() != 1 || bias.Shape()[0] != f {
			return nil, fmt.Errorf("ref: conv2d bias must be [%d], got %v", f, bias.Shape())
		}
	}
	s := attrs.Int("stride", 1)
	p := attrs.Int("pad", 0)
	if s < 1 || p < 0 {
		return nil, fmt.Errorf("ref: conv2d invalid stride %d / pad %d", s, p)
	}
	ho := (h+2*p-kh)/s + 1
	wo := (wd+2*p-kw)/s + 1
	if ho < 1 || wo < 1 {
		return nil, fmt.Errorf("ref: conv2d output would be empty (%dx%d)", ho, wo)
	}

	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, f, ho, wo})
	for ni := range n {
		for fi := range f {
			var b float64
			if bias != nil {
				b = bias.AtF64(fi)
			}
			for oy := range ho {
				for ox := range wo {
					// products in (c,ky,kx) order, bias added LAST — the same
					// linear accumulation order the im2col+GEMM path produces,
					// keeping cpu-vs-ref bit-identical (§V11 tol 0, §T24b)
					acc := 0.0
					for ci := range c {
						for ky := range kh {
							iy := oy*s + ky - p
							if iy < 0 || iy >= h {
								continue // zero padding
							}
							for kx := range kw {
								ix := ox*s + kx - p
								if ix < 0 || ix >= wd {
									continue
								}
								acc += x.AtF64(ni, ci, iy, ix) * w.AtF64(fi, ci, ky, kx)
							}
						}
					}
					out.SetF64(acc+b, ni, fi, oy, ox)
				}
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

// poolDims validates pooling geometry. Attrs: "kernel" (required), "stride"
// (default = kernel). No pool padding (torch golden generated accordingly).
func poolDims(x *tensor.Tensor, attrs backend.Attrs) (n, c, h, w, k, s, ho, wo int, err error) {
	if x.Ndim() != 4 {
		return 0, 0, 0, 0, 0, 0, 0, 0, fmt.Errorf("ref: pool needs x[N,C,H,W], got %v", x.Shape())
	}
	n, c, h, w = x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	k = attrs.Int("kernel", 0)
	if k < 1 {
		return 0, 0, 0, 0, 0, 0, 0, 0, fmt.Errorf("ref: pool kernel must be ≥1, got %d", k)
	}
	s = attrs.Int("stride", k)
	if s < 1 {
		return 0, 0, 0, 0, 0, 0, 0, 0, fmt.Errorf("ref: pool stride must be ≥1, got %d", s)
	}
	ho = (h-k)/s + 1
	wo = (w-k)/s + 1
	if ho < 1 || wo < 1 {
		return 0, 0, 0, 0, 0, 0, 0, 0, fmt.Errorf("ref: pool output would be empty")
	}
	return n, c, h, w, k, s, ho, wo, nil
}

func maxPool2dKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: maxpool2d wants 1 input, got %d", len(in))
	}
	x := in[0]
	n, c, _, _, k, s, ho, wo, err := poolDims(x, attrs)
	if err != nil {
		return nil, err
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, c, ho, wo})
	for ni := range n {
		for ci := range c {
			for oy := range ho {
				for ox := range wo {
					best := math.Inf(-1)
					for ky := range k {
						for kx := range k {
							if v := x.AtF64(ni, ci, oy*s+ky, ox*s+kx); v > best {
								best = v
							}
						}
					}
					out.SetF64(best, ni, ci, oy, ox)
				}
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func avgPool2dKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: avgpool2d wants 1 input, got %d", len(in))
	}
	x := in[0]
	n, c, _, _, k, s, ho, wo, err := poolDims(x, attrs)
	if err != nil {
		return nil, err
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, c, ho, wo})
	inv := 1 / float64(k*k)
	for ni := range n {
		for ci := range c {
			for oy := range ho {
				for ox := range wo {
					var acc float64
					for ky := range k {
						for kx := range k {
							acc += x.AtF64(ni, ci, oy*s+ky, ox*s+kx)
						}
					}
					out.SetF64(acc*inv, ni, ci, oy, ox)
				}
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpConv2D, conv2dKernel)
	reg(backend.OpMaxPool2D, maxPool2dKernel)
	reg(backend.OpAvgPool2D, avgPool2dKernel)
}
