package cpu

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// trailingReduce reports whether the reduction described by pa over a rank-nd shape reduces exactly
// the innermost CONTIGUOUS suffix of axes (e.g. axis 1 of [rows,cols], or axes {1,2} of [b,h,w]),
// with the reduced run longer than one element. When it does, each output value is the reduction of
// one contiguous segment xs[seg*segLen:(seg+1)*segLen] of the row-major input — so the per-op
// devirtualised reduce-all closure can be applied per segment, in parallel, instead of the reference
// kernel's per-element indirect combine. Returns the segment length, the (keepdims-aware) output
// shape, and ok. Multi-axis, strided, keepdims, and non-suffix reductions leave ok=false.
func trailingReduce(shape tensor.Shape, pa backend.ReduceAttrs) (segLen int, outShape tensor.Shape, ok bool) {
	nd := len(shape)
	if nd == 0 || len(pa.Axes) == 0 {
		return 0, nil, false // rank-0 or reduce-all: the reduce-all fast path already handles it
	}
	reduced := make([]bool, nd)
	for _, a := range pa.Axes {
		if a < 0 {
			a += nd
		}
		if a < 0 || a >= nd {
			return 0, nil, false // out of range: let the reference kernel report the error
		}
		reduced[a] = true
	}
	first := -1
	for ax := 0; ax < nd; ax++ {
		if reduced[ax] {
			first = ax
			break
		}
	}
	if first < 0 {
		return 0, nil, false
	}
	for ax := first; ax < nd; ax++ { // the reduced axes must run unbroken to the end (a kept axis after breaks the suffix)
		if !reduced[ax] {
			return 0, nil, false
		}
	}
	segLen = 1
	for ax := first; ax < nd; ax++ {
		segLen *= shape[ax]
	}
	if segLen <= 1 {
		return 0, nil, false // nothing to accumulate; ref handles the degenerate case
	}
	for ax := 0; ax < nd; ax++ {
		switch {
		case reduced[ax] && pa.KeepDims:
			outShape = append(outShape, 1)
		case reduced[ax]:
			// dropped
		default:
			outShape = append(outShape, shape[ax])
		}
	}
	return segLen, outShape, true
}

// leadingReduce reports whether the reduction reduces exactly a CONTIGUOUS PREFIX of axes that
// leaves a non-empty kept suffix (e.g. axis 0 of [rows,cols] → [cols], or axes {0,1} of [a,b,c] →
// [c]). When it does, the row-major input is a [numGroups, segStride] matrix and each output value
// output[p] folds the strided column xs[p], xs[segStride+p], … — which the reference kernel walks
// column-major (one cache line per element, the whole matrix re-streamed segStride times over). The
// leading fast path instead accumulates row-major into a segStride-wide accumulator: one pass over
// the input, contiguous reads, the accumulator cache-resident. Returns numGroups (the reduced
// prefix product, >1), segStride (the kept suffix product), the keepdims-aware output shape, and ok.
// The all-suffix case (reduced axes reach the last axis) is left to trailingReduce / reduce-all.
func leadingReduce(shape tensor.Shape, pa backend.ReduceAttrs) (numGroups, segStride int, outShape tensor.Shape, ok bool) {
	nd := len(shape)
	if nd == 0 || len(pa.Axes) == 0 {
		return 0, 0, nil, false
	}
	reduced := make([]bool, nd)
	for _, a := range pa.Axes {
		if a < 0 {
			a += nd
		}
		if a < 0 || a >= nd {
			return 0, 0, nil, false
		}
		reduced[a] = true
	}
	last := -1
	for ax := nd - 1; ax >= 0; ax-- {
		if reduced[ax] {
			last = ax
			break
		}
	}
	// The reduced axes must be exactly the prefix [0,last]; a kept axis inside the prefix
	// breaks contiguity, and last==nd-1 means the run reaches the end (trailing/reduce-all).
	if last < 0 || last == nd-1 {
		return 0, 0, nil, false
	}
	for ax := 0; ax <= last; ax++ {
		if !reduced[ax] {
			return 0, 0, nil, false
		}
	}
	numGroups, segStride = 1, 1
	for ax := 0; ax <= last; ax++ {
		numGroups *= shape[ax]
	}
	for ax := last + 1; ax < nd; ax++ {
		segStride *= shape[ax]
	}
	if numGroups <= 1 || segStride < 1 {
		return 0, 0, nil, false
	}
	for ax := 0; ax < nd; ax++ {
		switch {
		case reduced[ax] && pa.KeepDims:
			outShape = append(outShape, 1)
		case reduced[ax]:
			// dropped
		default:
			outShape = append(outShape, shape[ax])
		}
	}
	return numGroups, segStride, outShape, true
}

// leadReducer carries the per-op element fold for the leading-axis path: init is the fold
// identity, comb folds one element into the running accumulator (acc always f64, so the f32 input
// promotes), and mean divides by the reduced count at the end. The fold order is ascending group
// index per column — identical to the reference kernel — so Sum/Mean/Prod are bit-for-bit and
// Max/Min match by associativity+commutativity of the IEEE min/max used on both sides.
type leadReducer struct {
	init float64
	comb func(acc, x float64) float64
	mean bool
}

// wrapReduceAll returns a CPU kernel that fast-paths the reduce-ALL-to-scalar case (attrs
// nil or empty Axes) with a devirtualised, per-op INLINED loop over the contiguous storage.
// The reference reduceKernel is correct but runs single-threaded through a per-element
// `combine` func value — an indirect call the compiler cannot inline or vectorise — plus a
// per-element odometer that is pure overhead when the output never moves (reduce-all). Here
// f64/f32 is the passed loop closure, called ONCE per invocation, so its inner accumulation
// inlines. Axis reductions, keepdims edge cases, and exotic dtypes fall back to the ref
// kernel unchanged. Bit-identical: same ascending row-major order, same init/combine/finalize.
func wrapReduceAll(op backend.Op, f64 func([]float64) float64, f32 func([]float32) float64, lr leadReducer) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) == 1 {
			pa, _ := attrs.(backend.ReduceAttrs)
			x := in[0]
			if dt := x.Dtype(); dt == tensor.F64 || dt == tensor.F32 {
				if attrs == nil || len(pa.Axes) == 0 {
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
				// Trailing-contiguous axis reduction: each output is one contiguous segment, so the
				// SAME devirtualised reduce-all closure (inlined Sum/Mean, 4-way-unrolled Max/Min)
				// runs per segment in parallel — dropping the reference kernel's per-element indirect
				// combine call (the latency-bound math.Max/math.Min chain in particular). Bit-identical:
				// each segment sees xs[seg*segLen:(seg+1)*segLen] in the same ascending row-major order
				// the reference odometer visits, and the closure's fold matches the ref per §T-reduce.
				if segLen, outShape, ok := trailingReduce(x.Shape(), pa); ok {
					xc := x.Contiguous()
					out := tensor.NewOn(ctx.Device(), dt, outShape)
					outNumel := outShape.Numel()
					if dt == tensor.F64 {
						xs, os := xc.Storage().F64(), out.Storage().F64()
						parallelWork(outNumel, segLen, func(lo, hi int) {
							for seg := lo; seg < hi; seg++ {
								os[seg] = f64(xs[seg*segLen : (seg+1)*segLen])
							}
						})
					} else {
						xs, os := xc.Storage().F32(), out.Storage().F32()
						parallelWork(outNumel, segLen, func(lo, hi int) {
							for seg := lo; seg < hi; seg++ {
								os[seg] = float32(f32(xs[seg*segLen : (seg+1)*segLen]))
							}
						})
					}
					return []*tensor.Tensor{out}, nil
				}
				// Leading-contiguous (prefix) axis reduction: the input is a [numGroups, segStride]
				// row-major matrix and each output column folds the strided values one per group. The
				// reference walks it column-major (the whole matrix re-streamed segStride times); this
				// folds row-major into a segStride-wide accumulator (one pass, contiguous reads, acc
				// cache-resident). Parallel over disjoint output columns [plo,phi) — each column's fold
				// is independent, ascending-group order preserved, so bit-identical to the reference.
				if lr.comb != nil {
					if numGroups, segStride, outShape, ok := leadingReduce(x.Shape(), pa); ok {
						xc := x.Contiguous()
						out := tensor.NewOn(ctx.Device(), dt, outShape)
						comb, initV, mean := lr.comb, lr.init, lr.mean
						invN := 1.0 / float64(numGroups)
						if dt == tensor.F64 {
							xs, os := xc.Storage().F64(), out.Storage().F64()
							parallelWork(segStride, numGroups, func(plo, phi int) {
								for p := plo; p < phi; p++ {
									os[p] = initV
								}
								for g := 0; g < numGroups; g++ {
									base := g * segStride
									for p := plo; p < phi; p++ {
										os[p] = comb(os[p], xs[base+p])
									}
								}
								if mean {
									for p := plo; p < phi; p++ {
										os[p] *= invN
									}
								}
							})
						} else {
							xs, os := xc.Storage().F32(), out.Storage().F32()
							parallelWork(segStride, numGroups, func(plo, phi int) {
								acc := make([]float64, phi-plo)
								for i := range acc {
									acc[i] = initV
								}
								for g := 0; g < numGroups; g++ {
									base := g * segStride
									for p := plo; p < phi; p++ {
										acc[p-plo] = comb(acc[p-plo], float64(xs[base+p]))
									}
								}
								for p := plo; p < phi; p++ {
									v := acc[p-plo]
									if mean {
										v *= invN
									}
									os[p] = float32(v)
								}
							})
						}
						return []*tensor.Tensor{out}, nil
					}
				}
			}
		}
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
	}
}

func init() {
	reg := func(op backend.Op, f64 func([]float64) float64, f32 func([]float32) float64, lr leadReducer) {
		k := wrapReduceAll(op, f64, f32, lr)
		//perfscan:ignore PS3052 false-positive: kernel registration init, one-time
		std.add(op, tensor.F64, k)
		std.add(op, tensor.F32, k)
	}
	add := func(a, x float64) float64 { return a + x }
	mul := func(a, x float64) float64 { return a * x }
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
		},
		leadReducer{init: 0, comb: add})
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
		},
		leadReducer{init: 0, comb: add, mean: true})
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
				//perfscan:ignore PS3019 deliberate 4-way Max unroll, already optimal fastpath
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
				//perfscan:ignore PS3019 deliberate 4-way Max f32 unroll, already optimal
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
		},
		leadReducer{init: math.Inf(-1), comb: math.Max})
	reg(backend.OpMin,
		func(xs []float64) float64 {
			m0, m1, m2, m3 := math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)
			i := 0
			for ; i+4 <= len(xs); i += 4 {
				//perfscan:ignore PS3019 deliberate 4-way Min unroll, already optimal fastpath
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
				//perfscan:ignore PS3019 deliberate 4-way Min f32 unroll, already optimal
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
		},
		leadReducer{init: math.Inf(1), comb: math.Min})
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
		},
		leadReducer{init: 1, comb: mul})
}
