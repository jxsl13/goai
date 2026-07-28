package cpu

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// wrapReduceAll returns a CPU kernel that fast-paths the reduce-ALL-to-scalar case (attrs
// nil or empty Axes) with a devirtualised, per-op INLINED loop over the contiguous storage.
// The reference reduceKernel is correct but runs single-threaded through a per-element
// `combine` func value — an indirect call the compiler cannot inline or vectorise — plus a
// per-element odometer that is pure overhead when the output never moves (reduce-all). Here
// f64/f32 is the passed loop closure, called ONCE per invocation, so its inner accumulation
// inlines. Axis reductions, keepdims edge cases, and exotic dtypes fall back to the ref
// kernel unchanged. Bit-identical: same ascending row-major order, same init/combine/finalize.
func wrapReduceAll(op backend.Op, f64 func([]float64) float64, f32 func([]float32) float64) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) == 1 {
			pa, _ := attrs.(backend.ReduceAttrs)
			if attrs == nil || len(pa.Axes) == 0 {
				x := in[0]
				if dt := x.Dtype(); dt == tensor.F64 || dt == tensor.F32 {
					xc := x.Contiguous()
					var acc float64
					if dt == tensor.F64 {
						acc = f64(xc.Storage().F64())
					} else {
						acc = f32(xc.Storage().F32())
					}
					outShape := tensor.Shape{}
					if pa.KeepDims {
						outShape = make(tensor.Shape, x.Ndim())
						for i := range outShape {
							outShape[i] = 1
						}
					}
					out := tensor.NewOn(ctx.Device(), dt, outShape)
					out.SetF64(acc, make([]int, len(outShape))...)
					return []*tensor.Tensor{out}, nil
				}
			}
		}
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
	}
}

func init() {
	reg := func(op backend.Op, f64 func([]float64) float64, f32 func([]float32) float64) {
		k := wrapReduceAll(op, f64, f32)
		std.add(op, tensor.F64, k)
		std.add(op, tensor.F32, k)
	}
	reg(backend.OpSum,
		func(xs []float64) float64 {
			var s float64
			for _, v := range xs {
				s += v
			}
			return s
		},
		func(xs []float32) float64 {
			var s float64
			for _, v := range xs {
				s += float64(v)
			}
			return s
		})
	reg(backend.OpMean,
		func(xs []float64) float64 {
			var s float64
			for _, v := range xs {
				s += v
			}
			return s / float64(len(xs))
		},
		func(xs []float32) float64 {
			var s float64
			for _, v := range xs {
				s += float64(v)
			}
			return s / float64(len(xs))
		})
	reg(backend.OpMax,
		func(xs []float64) float64 {
			m := math.Inf(-1)
			for _, v := range xs {
				m = math.Max(m, v)
			}
			return m
		},
		func(xs []float32) float64 {
			m := math.Inf(-1)
			for _, v := range xs {
				m = math.Max(m, float64(v))
			}
			return m
		})
	reg(backend.OpMin,
		func(xs []float64) float64 {
			m := math.Inf(1)
			for _, v := range xs {
				m = math.Min(m, v)
			}
			return m
		},
		func(xs []float32) float64 {
			m := math.Inf(1)
			for _, v := range xs {
				m = math.Min(m, float64(v))
			}
			return m
		})
	reg(backend.OpProd,
		func(xs []float64) float64 {
			p := 1.0
			for _, v := range xs {
				p *= v
			}
			return p
		},
		func(xs []float32) float64 {
			p := 1.0
			for _, v := range xs {
				p *= float64(v)
			}
			return p
		})
}
