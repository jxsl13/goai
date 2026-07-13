package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized 2-D pooling (§T462): the reference kernels walk AtF64 per element;
// these run on contiguous storage with the channel planes parallelized. Window
// iteration order and arithmetic match the reference exactly (max via strict >
// from −Inf; avg accumulates in (ky,kx) order, one multiply by 1/k²), so results
// are bit-identical (§V3, §V11 tol 0). Also serves every other backend's missing
// pool via the §T461 fallback chain.

func poolDimsCPU(x *tensor.Tensor, attrs backend.Attrs) (n, c, h, w, k, s, ho, wo int, err error) {
	if x.Ndim() != 4 {
		return 0, 0, 0, 0, 0, 0, 0, 0, fmt.Errorf("cpu: pool needs x[N,C,H,W], got %v", x.Shape())
	}
	pa, _ := attrs.(backend.PoolAttrs)
	pa = pa.WithDefaults()
	k, s = pa.Kernel, pa.Stride
	n, c, h, w = x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	if k < 1 || s < 1 {
		return 0, 0, 0, 0, 0, 0, 0, 0, fmt.Errorf("cpu: pool invalid kernel %d / stride %d", k, s)
	}
	ho, wo = (h-k)/s+1, (w-k)/s+1
	if h < k || w < k || ho < 1 || wo < 1 {
		return 0, 0, 0, 0, 0, 0, 0, 0, fmt.Errorf("cpu: pool window %d exceeds input %dx%d", k, h, w)
	}
	return n, c, h, w, k, s, ho, wo, nil
}

func pool2dKernel(isMax bool) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) != 1 {
			return nil, fmt.Errorf("cpu: pool wants 1 input, got %d", len(in))
		}
		x := in[0]
		n, c, h, w, k, s, ho, wo, err := poolDimsCPU(x, attrs)
		if err != nil {
			return nil, err
		}
		xc := x.Contiguous()
		out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, c, ho, wo})
		inv := 1 / float64(k*k)

		run := func(get func(flat int) float64, set func(flat int, v float64)) {
			planes := n * c
			parallelWork(planes, ho*wo*k*k, func(lo, hi int) {
				for pl := lo; pl < hi; pl++ {
					in0 := pl * h * w
					out0 := pl * ho * wo
					for oy := range ho {
						for ox := range wo {
							if isMax {
								best := math.Inf(-1)
								for ky := range k {
									row := in0 + (oy*s+ky)*w + ox*s
									for kx := range k {
										if v := get(row + kx); v > best {
											best = v
										}
									}
								}
								set(out0+oy*wo+ox, best)
							} else {
								var acc float64
								for ky := range k {
									row := in0 + (oy*s+ky)*w + ox*s
									for kx := range k {
										acc += get(row + kx)
									}
								}
								set(out0+oy*wo+ox, acc*inv)
							}
						}
					}
				}
			})
		}
		switch x.Dtype() {
		case tensor.F64:
			xs, os := xc.Storage().F64(), out.Storage().F64()
			run(func(i int) float64 { return xs[i] }, func(i int, v float64) { os[i] = v })
		case tensor.F32:
			xs, os := xc.Storage().F32(), out.Storage().F32()
			run(func(i int) float64 { return float64(xs[i]) }, func(i int, v float64) { os[i] = float32(v) })
		default:
			return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
		}
		return []*tensor.Tensor{out}, nil
	}
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpMaxPool2D, dt, pool2dKernel(true))
		std.add(backend.OpAvgPool2D, dt, pool2dKernel(false))
	}
}
