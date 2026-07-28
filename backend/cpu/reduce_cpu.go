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
	// Max/Min use a 4-way accumulator unroll to break the loop-carried dependency
	// chain. math.Max/math.Min are NOT intrinsified on amd64 (hardware MAXSD/MINSD
	// disagree on NaN/signed-zero), so each element is a real non-inlined CALL whose
	// result feeds the next — the serial single-accumulator form is latency-bound.
	// Four independent chains let the calls overlap (throughput-bound). Bit-exact
	// (unlike Sum/Prod reassociation): math.Max/math.Min are associative AND
	// commutative including their IEEE tie rules — NaN is absorbing, and
	// math.Max(+0,-0)=+0 / math.Min(+0,-0)=-0 regardless of grouping — so folding the
	// four partials yields the same bits as the ascending serial scan (and the ref
	// kernel, which uses the same math.Max/math.Min with ±Inf init).
	reg(backend.OpMax,
		func(xs []float64) float64 {
			m0, m1, m2, m3 := math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1)
			i := 0
			for ; i+4 <= len(xs); i += 4 {
				m0 = math.Max(m0, xs[i])
				m1 = math.Max(m1, xs[i+1])
				m2 = math.Max(m2, xs[i+2])
				m3 = math.Max(m3, xs[i+3])
			}
			m := math.Max(math.Max(m0, m1), math.Max(m2, m3))
			for ; i < len(xs); i++ {
				m = math.Max(m, xs[i])
			}
			return m
		},
		func(xs []float32) float64 {
			m0, m1, m2, m3 := math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1)
			i := 0
			for ; i+4 <= len(xs); i += 4 {
				m0 = math.Max(m0, float64(xs[i]))
				m1 = math.Max(m1, float64(xs[i+1]))
				m2 = math.Max(m2, float64(xs[i+2]))
				m3 = math.Max(m3, float64(xs[i+3]))
			}
			m := math.Max(math.Max(m0, m1), math.Max(m2, m3))
			for ; i < len(xs); i++ {
				m = math.Max(m, float64(xs[i]))
			}
			return m
		})
	reg(backend.OpMin,
		func(xs []float64) float64 {
			m0, m1, m2, m3 := math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)
			i := 0
			for ; i+4 <= len(xs); i += 4 {
				m0 = math.Min(m0, xs[i])
				m1 = math.Min(m1, xs[i+1])
				m2 = math.Min(m2, xs[i+2])
				m3 = math.Min(m3, xs[i+3])
			}
			m := math.Min(math.Min(m0, m1), math.Min(m2, m3))
			for ; i < len(xs); i++ {
				m = math.Min(m, xs[i])
			}
			return m
		},
		func(xs []float32) float64 {
			m0, m1, m2, m3 := math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)
			i := 0
			for ; i+4 <= len(xs); i += 4 {
				m0 = math.Min(m0, float64(xs[i]))
				m1 = math.Min(m1, float64(xs[i+1]))
				m2 = math.Min(m2, float64(xs[i+2]))
				m3 = math.Min(m3, float64(xs[i+3]))
			}
			m := math.Min(math.Min(m0, m1), math.Min(m2, m3))
			for ; i < len(xs); i++ {
				m = math.Min(m, float64(xs[i]))
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
