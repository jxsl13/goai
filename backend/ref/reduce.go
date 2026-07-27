package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Reduction reference kernels (§T7). Accumulation is always in float64 even for
// F32 inputs, narrowing only the final result — this is §V10 ACCUM, which keeps
// f32 sums from drifting with large reduction length. Attrs: "axes" []int
// (absent → reduce all), "keepdims" bool. argmax uses "axis" int (absent → flat).

// reduceAllAxis marks "no axis given" for argmax.
const reduceAllAxis = math.MinInt

// reducedShape computes the output shape and a map from input axis → output-axis
// position (-1 = omitted). reduced[ax] marks axes being reduced.
func reducedShape(in tensor.Shape, reduced []bool, keepdims bool) (tensor.Shape, []int) {
	nd := len(in)
	out := tensor.Shape{}
	outAxisOf := make([]int, nd)
	k := 0
	for ax := range nd {
		switch {
		case reduced[ax] && keepdims:
			out = append(out, 1)
			outAxisOf[ax] = k
			k++
		case reduced[ax]:
			outAxisOf[ax] = -1
		default:
			out = append(out, in[ax])
			outAxisOf[ax] = k
			k++
		}
	}
	return out, outAxisOf
}

func parseReducedMask(attrs backend.Attrs, nd int) ([]bool, error) {
	reduced := make([]bool, nd)
	pa, _ := attrs.(backend.ReduceAttrs)
	axes := pa.Axes
	if len(axes) == 0 {
		for i := range reduced {
			reduced[i] = true
		}
		return reduced, nil
	}
	for _, a := range axes {
		if a < 0 {
			a += nd
		}
		if a < 0 || a >= nd {
			return nil, fmt.Errorf("ref: reduce axis %d out of range for rank %d", a, nd)
		}
		reduced[a] = true
	}
	return reduced, nil
}

func reduceKernel(init float64, combine func(acc, x float64) float64, finalize func(acc float64, count int) float64) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) != 1 {
			return nil, fmt.Errorf("ref: reduce wants 1 input, got %d", len(in))
		}
		x := in[0]
		nd := x.Ndim()
		reduced, err := parseReducedMask(attrs, nd)
		if err != nil {
			return nil, err
		}
		pa, _ := attrs.(backend.ReduceAttrs)
		keepdims := pa.KeepDims
		outShape, outAxisOf := reducedShape(x.Shape(), reduced, keepdims)
		outStrides := tensor.RowMajorStrides(outShape)
		outNumel := outShape.Numel()

		acc := make([]float64, outNumel)
		for i := range acc {
			acc[i] = init
		}
		count := 1
		if outNumel > 0 {
			count = x.Numel() / outNumel // product of reduced-axis sizes
		}
		// Devirtualised traversal (§T646 follow-up): the generic loop pays a
		// per-element Unravel heap alloc + AtF64 dispatch + an O(nd) offset
		// rebuild. Here the input is exposed once as a flat row-major []float64
		// (exact widening for F32) and the output offset is carried
		// INCREMENTALLY by an odometer over per-axis effective strides
		// (eff[ax] = outStrides[outAxisOf[ax]], 0 for reduced axes — exactly the
		// coord·stride sum of the generic loop). Element order is the same
		// ascending row-major pos, so each accumulator sees the same combine
		// sequence — bit-identical.
		if xs, ok := f64Data(x); ok {
			shape := x.Shape()
			eff := make([]int, nd)
			for ax := range nd {
				if p := outAxisOf[ax]; p >= 0 && !reduced[ax] {
					eff[ax] = outStrides[p]
				}
			}
			if len(acc) == 1 {
				// Reduce-all: every element lands in acc[0], so `of` is invariably 0
				// and the odometer is pure bookkeeping — for this shape every eff is
				// zero, so it computes of += 0 once per element. A local accumulator
				// additionally removes the acc[0] load/store and its bounds check from
				// the inner loop. Same ascending pos order, same combine sequence, so
				// the accumulator sees an identical add chain — bit-identical.
				a := acc[0]
				for _, v := range xs {
					a = combine(a, v)
				}
				acc[0] = a
			} else if nd > 0 && shape[nd-1] > 0 {
				// Hoist the innermost axis out of the odometer. Its effective stride is
				// constant across the run, so the run is either a single accumulator
				// fed repeatedly (stride 0 — a reduced innermost axis) or a straight
				// walk down consecutive accumulators (stride 1 — the innermost axis
				// survives into the output). Either way the odometer ticks once per run
				// instead of once per element.
				//
				// Bit-identical: every accumulator still sees exactly the same values in
				// the same ascending order, so its combine chain is unchanged. Only the
				// index bookkeeping around it moves.
				inner := shape[nd-1]
				sInner := eff[nd-1]
				idx := make([]int, nd)
				of, pos := 0, 0
				for pos < len(xs) {
					run := xs[pos : pos+inner]
					switch sInner {
					case 0: // whole run folds into one accumulator
						a := acc[of]
						for _, v := range run {
							a = combine(a, v)
						}
						acc[of] = a
					case 1: // run walks consecutive accumulators
						dst := acc[of : of+inner]
						for j, v := range run {
							dst[j] = combine(dst[j], v)
						}
					default: // strided (not reachable for row-major outputs; kept total)
						o := of
						for _, v := range run {
							acc[o] = combine(acc[o], v)
							o += sInner
						}
					}
					pos += inner
					// The innermost axis completed, so its net contribution to `of` is
					// zero. Tick the remaining axes exactly as the per-element odometer
					// would have.
					for d := nd - 2; d >= 0; d-- {
						idx[d]++
						of += eff[d]
						if idx[d] < shape[d] {
							break
						}
						idx[d] = 0
						of -= eff[d] * shape[d]
					}
				}
			} else {
				idx := make([]int, nd)
				of := 0
				for pos := range xs {
					acc[of] = combine(acc[of], xs[pos])
					for d := nd - 1; d >= 0; d-- {
						idx[d]++
						of += eff[d]
						if idx[d] < shape[d] {
							break
						}
						idx[d] = 0
						of -= eff[d] * shape[d]
					}
				}
			}
			out := tensor.NewOn(ctx.Device(), x.Dtype(), outShape)
			os, flush, _ := outF64(out) // dtype is F32/F64 here (f64Data ok), so outF64 cannot fail
			for i := range acc {
				os[i] = finalize(acc[i], count)
			}
			flush()
			return []*tensor.Tensor{out}, nil
		}
		// Generic fallback for exotic dtypes (verbatim original loop).
		for pos := range x.Numel() {
			idx := tensor.Unravel(pos, x.Shape())
			of := 0
			for ax := range nd {
				p := outAxisOf[ax]
				if p < 0 {
					continue
				}
				coord := idx[ax]
				if reduced[ax] { // keepdims reduced axis → coord 0
					coord = 0
				}
				of += coord * outStrides[p]
			}
			acc[of] = combine(acc[of], x.AtF64(idx...))
		}

		out := tensor.NewOn(ctx.Device(), x.Dtype(), outShape)
		for i := range acc {
			out.SetF64(finalize(acc[i], count), tensor.Unravel(i, outShape)...)
		}
		return []*tensor.Tensor{out}, nil
	}
}

// argmaxKernel returns the index of the maximum along "axis" (absent → flat index
// over all elements). Ties resolve to the lowest index (numpy semantics). Output
// holds the index as a float in the input dtype (interim until int dtypes, §B12).
func argmaxKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: argmax wants 1 input, got %d", len(in))
	}
	x := in[0]
	nd := x.Ndim()
	axis := reduceAllAxis
	if pa, ok := attrs.(backend.ArgMaxAttrs); ok {
		axis = pa.Axis
	}

	if axis == reduceAllAxis {
		best := math.Inf(-1)
		bi := 0
		// Devirtualised flat scan (§T646 follow-up): same ascending-pos order and
		// strict > comparison as the generic loop — bit-identical tie handling.
		if xs, ok := f64Data(x); ok {
			for pos, v := range xs {
				if v > best {
					best, bi = v, pos
				}
			}
		} else {
			for pos := range x.Numel() {
				v := x.AtF64(tensor.Unravel(pos, x.Shape())...)
				if v > best {
					best, bi = v, pos
				}
			}
		}
		out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{})
		out.SetF64(float64(bi))
		return []*tensor.Tensor{out}, nil
	}

	if axis < 0 {
		axis += nd
	}
	if axis < 0 || axis >= nd {
		return nil, fmt.Errorf("ref: argmax axis %d out of range for rank %d", axis, nd)
	}
	reduced := make([]bool, nd)
	reduced[axis] = true
	outShape, outAxisOf := reducedShape(x.Shape(), reduced, false)
	outStrides := tensor.RowMajorStrides(outShape)
	outNumel := outShape.Numel()

	best := make([]float64, outNumel)
	bidx := make([]float64, outNumel)
	for i := range best {
		best[i] = math.Inf(-1)
	}
	// Devirtualised traversal (§T646 follow-up): flat typed input + an odometer
	// carrying the output offset incrementally (see reduceKernel). Same
	// ascending-pos order and strict > comparison — bit-identical tie handling.
	if xs, ok := f64Data(x); ok {
		shape := x.Shape()
		eff := make([]int, nd)
		for ax := range nd {
			if p := outAxisOf[ax]; p >= 0 {
				eff[ax] = outStrides[p]
			}
		}
		// Hoist the innermost axis out of the odometer (see reduceKernel): its
		// effective stride is constant across a run, so the odometer ticks once per
		// run instead of once per element. Two cases, because argmax carries a
		// COORDINATE as well as a value:
		//   - the innermost axis IS the reduced axis: its stride is 0, the whole run
		//     folds into one accumulator, and the coordinate is the position within
		//     the run;
		//   - otherwise: the reduced coordinate is constant across the run, and the
		//     run walks consecutive accumulators.
		// Both keep the ascending scan order and the STRICT > comparison, so ties
		// still resolve to the lowest index — bit-identical.
		inner, sInner := shape[nd-1], eff[nd-1]
		axisIsInnermost := axis == nd-1
		idx := make([]int, nd)
		of := 0
		for pos := 0; pos < len(xs); pos += inner {
			run := xs[pos : pos+inner]
			if axisIsInnermost {
				b, bi := best[of], bidx[of]
				for j, v := range run {
					if v > b {
						b, bi = v, float64(j)
					}
				}
				best[of], bidx[of] = b, bi
			} else {
				c := float64(idx[axis])
				o := of
				for _, v := range run {
					if v > best[o] {
						best[o] = v
						bidx[o] = c
					}
					o += sInner
				}
			}
			for d := nd - 2; d >= 0; d-- {
				idx[d]++
				of += eff[d]
				if idx[d] < shape[d] {
					break
				}
				idx[d] = 0
				of -= eff[d] * shape[d]
			}
		}
		out := tensor.NewOn(ctx.Device(), x.Dtype(), outShape)
		os, flush, _ := outF64(out) // dtype is F32/F64 here, outF64 cannot fail
		copy(os, bidx)
		flush()
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for pos := range x.Numel() {
		idx := tensor.Unravel(pos, x.Shape())
		of := 0
		for ax := range nd {
			if p := outAxisOf[ax]; p >= 0 {
				of += idx[ax] * outStrides[p]
			}
		}
		if v := x.AtF64(idx...); v > best[of] {
			best[of] = v
			bidx[of] = float64(idx[axis])
		}
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), outShape)
	for i := range bidx {
		out.SetF64(bidx[i], tensor.Unravel(i, outShape)...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpSum, reduceKernel(0, func(a, x float64) float64 { return a + x }, func(a float64, _ int) float64 { return a }))
	reg(backend.OpMean, reduceKernel(0, func(a, x float64) float64 { return a + x }, func(a float64, n int) float64 { return a / float64(n) }))
	reg(backend.OpMax, reduceKernel(math.Inf(-1), math.Max, func(a float64, _ int) float64 { return a }))
	reg(backend.OpMin, reduceKernel(math.Inf(1), math.Min, func(a float64, _ int) float64 { return a }))
	reg(backend.OpProd, reduceKernel(1, func(a, x float64) float64 { return a * x }, func(a float64, _ int) float64 { return a }))
	reg(backend.OpArgMax, argmaxKernel)
}
