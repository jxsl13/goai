package autograd

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Reduction VJPs (§T14). The gradient of a reduction broadcasts g back over the
// reduced axes: sum copies, mean scales by 1/count, max/min route to the FIRST
// maximal/minimal element per group (tie → lowest index, matching the argmax
// tie rule, §B16). argmax itself is non-differentiable → nil grad.

// reduceOutMap mirrors the reference reduction geometry: it returns the output
// shape and a function mapping an input multi-index to the flat output offset,
// honoring the "axes" and "keepdims" attrs.
func reduceOutMap(xShape tensor.Shape, attrs backend.Attrs) (tensor.Shape, func(idx []int) int, []int, error) {
	nd := len(xShape)
	reduced := make([]bool, nd)
	pr, _ := attrs.(backend.ReduceAttrs)
	axes := pr.Axes
	if len(axes) == 0 {
		for i := range reduced {
			reduced[i] = true
		}
	} else {
		for _, a := range axes {
			if a < 0 {
				a += nd
			}
			if a < 0 || a >= nd {
				return nil, nil, nil, fmt.Errorf("autograd: reduce axis %d out of range for rank %d", a, nd)
			}
			reduced[a] = true
		}
	}
	keepdims := pr.KeepDims

	outShape := tensor.Shape{}
	outAxisOf := make([]int, nd)
	k := 0
	for ax := range nd {
		switch {
		case reduced[ax] && keepdims:
			outShape = append(outShape, 1)
			outAxisOf[ax] = k
			k++
		case reduced[ax]:
			outAxisOf[ax] = -1
		default:
			outShape = append(outShape, xShape[ax])
			outAxisOf[ax] = k
			k++
		}
	}
	outStrides := tensor.RowMajorStrides(outShape)
	mapIdx := func(idx []int) int {
		of := 0
		for ax := range nd {
			p := outAxisOf[ax]
			if p < 0 {
				continue
			}
			coord := idx[ax]
			if reduced[ax] {
				coord = 0
			}
			of += coord * outStrides[p]
		}
		return of
	}
	// axStride[ax] = the change in the output flat offset per unit increment of input
	// axis ax (0 for reduced axes, since mapIdx pins their coord to 0). Lets callers
	// walk the input with an incremental offset instead of re-Unravel-ing per element.
	axStride := make([]int, nd)
	for ax := range nd {
		if !reduced[ax] && outAxisOf[ax] >= 0 {
			axStride[ax] = outStrides[outAxisOf[ax]]
		}
	}
	return outShape, mapIdx, axStride, nil
}

// broadcastVJP builds sum/mean gradients: gin[idx] = g[map(idx)] · scale(count).
func broadcastVJP(mean bool) VJP {
	return func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		outShape, _, axStride, err := reduceOutMap(x.Shape(), attrs)
		if err != nil {
			return nil, err
		}
		scale := 1.0
		if mean {
			if on := outShape.Numel(); on > 0 {
				scale = float64(on) / float64(x.Numel()) // 1/count
			}
		}
		gin := tensor.New(x.Dtype(), x.Shape())
		n := x.Numel()
		xs := x.Shape()
		nd := len(xs)
		gc := g.Contiguous()
		// walk x in row-major flat order, maintaining the output flat offset `of`
		// incrementally (odometer) — no per-element Unravel, no double index round-trip;
		// dtype-switch once for direct []T access (§base-perf, C25, §T633).
		coord := make([]int, nd)
		of := 0
		switch x.Dtype() {
		case tensor.F64:
			gs, ds := gc.Storage().F64(), gin.Storage().F64()
			if !broadcastFillRows(ds, gs, xs, axStride, n, scale) {
				// Declined-shape fallback: the per-element odometer is deliberate here.
				//perfscan:ignore PS4005 fallback for shapes broadcastFillRows declines
				for i := 0; i < n; i++ {
					ds[i] = gs[of] * scale
					for ax := nd - 1; ax >= 0; ax-- {
						coord[ax]++
						of += axStride[ax]
						if coord[ax] < xs[ax] {
							break
						}
						coord[ax] = 0
						of -= axStride[ax] * xs[ax]
					}
				}
			}
		case tensor.F32:
			gs, ds := gc.Storage().F32(), gin.Storage().F32()
			sc := float32(scale)
			if !broadcastFillRows(ds, gs, xs, axStride, n, sc) {
				// Declined-shape fallback: the per-element odometer is deliberate here.
				//perfscan:ignore PS4005 fallback for shapes broadcastFillRows declines
				for i := 0; i < n; i++ {
					ds[i] = gs[of] * sc
					for ax := nd - 1; ax >= 0; ax-- {
						coord[ax]++
						of += axStride[ax]
						if coord[ax] < xs[ax] {
							break
						}
						coord[ax] = 0
						of -= axStride[ax] * xs[ax]
					}
				}
			}
		default:
			mapIdx := func(c []int) int {
				o := 0
				for ax := 0; ax < nd; ax++ {
					o += c[ax] * axStride[ax]
				}
				return o
			}
			for i := 0; i < n; i++ {
				idx := tensor.Unravel(i, xs)
				gin.SetF64(gc.AtF64(tensor.Unravel(mapIdx(idx), outShape)...)*scale, idx...)
			}
		}
		return []*tensor.Tensor{gin}, nil
	}
}

// extremumVJP routes g to the first element attaining the group extremum (§B16).
func extremumVJP() VJP {
	return func(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		outShape, mapIdx, axStride, err := reduceOutMap(x.Shape(), attrs)
		if err != nil {
			return nil, err
		}
		gin := tensor.New(x.Dtype(), x.Shape())
		routed := make([]bool, outShape.Numel())
		n := x.Numel()
		xc, yc, gc := x.Contiguous(), y.Contiguous(), g.Contiguous()
		if x.Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
			xs, ys, gs, ds := xc.Storage().F64(), yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
			forEachReduceRow(x.Shape(), axStride, func(i0, of0, inner, sInner int) {
				if sInner == 0 {
					// of is constant across the row (the inner axis is reduced), so
					// route the first element equal to ys[of0] and stop — the rest of
					// the row is a no-op once routed[of0] is set. Same first-hit index
					// and same single write as the full scan; only the tail is skipped.
					if routed[of0] {
						return
					}
					yv := ys[of0]
					for p := 0; p < inner; p++ {
						if xs[i0+p] == yv {
							ds[i0+p] = gs[of0]
							routed[of0] = true
							return
						}
					}
					return
				}
				of := of0
				for p := 0; p < inner; p++ {
					i := i0 + p
					if !routed[of] && xs[i] == ys[of] {
						ds[i] = gs[of]
						routed[of] = true
					}
					of += sInner
				}
			})
			return []*tensor.Tensor{gin}, nil
		}
		if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
			xs, ys, gs, ds := xc.Storage().F32(), yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
			forEachReduceRow(x.Shape(), axStride, func(i0, of0, inner, sInner int) {
				if sInner == 0 {
					// of is constant across the row (the inner axis is reduced), so
					// route the first element equal to ys[of0] and stop — the rest of
					// the row is a no-op once routed[of0] is set. Same first-hit index
					// and same single write as the full scan; only the tail is skipped.
					if routed[of0] {
						return
					}
					yv := ys[of0]
					for p := 0; p < inner; p++ {
						if xs[i0+p] == yv {
							ds[i0+p] = gs[of0]
							routed[of0] = true
							return
						}
					}
					return
				}
				of := of0
				for p := 0; p < inner; p++ {
					i := i0 + p
					if !routed[of] && xs[i] == ys[of] {
						ds[i] = gs[of]
						routed[of] = true
					}
					of += sInner
				}
			})
			return []*tensor.Tensor{gin}, nil
		}
		for i := 0; i < n; i++ { // generic fallback (exotic dtype)
			idx := tensor.Unravel(i, x.Shape())
			of := mapIdx(idx)
			if routed[of] {
				continue
			}
			oidx := tensor.Unravel(of, outShape)
			if x.AtF64(idx...) == y.AtF64(oidx...) {
				gin.SetF64(g.AtF64(oidx...), idx...)
				routed[of] = true
			}
		}
		return []*tensor.Tensor{gin}, nil
	}
}

// prodVJP builds the product-reduction gradient: ∂prod/∂xᵢ = ∏_{j≠i} xⱼ (over the
// element's reduction group). Computed as prodₐₗₗ/xᵢ when the group has no zeros;
// when the group has exactly one zero, only that element gets the product of the
// other (nonzero) elements; with two or more zeros every gradient is 0.
func prodVJP() VJP {
	return func(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		outShape, mapIdx, axStride, err := reduceOutMap(x.Shape(), attrs)
		if err != nil {
			return nil, err
		}
		on := outShape.Numel()
		numZeros := make([]int, on)
		prodNz := make([]float64, on) // product of the nonzero elements per group
		for i := range prodNz {
			prodNz[i] = 1
		}
		gin := tensor.New(x.Dtype(), x.Shape())
		n := x.Numel()
		xc, yc, gc := x.Contiguous(), y.Contiguous(), g.Contiguous()
		// grad(xᵢ) = ∏_{j≠i} xⱼ: prodₐₗₗ/xᵢ (no zeros), or the other-elements product
		// for the lone zero (case 1), else 0. Typed two-pass with an incremental
		// output offset (prod accumulation stays f64 for precision regardless of dtype).
		if x.Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
			xs, ys, gs, ds := xc.Storage().F64(), yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
			// Pass 1: tally zeros and the non-zero product per group.
			forEachReduceRow(x.Shape(), axStride, func(i0, of0, inner, sInner int) {
				of := of0
				for p := 0; p < inner; p++ {
					i := i0 + p
					if xs[i] == 0 {
						numZeros[of]++
					} else {
						prodNz[of] *= xs[i]
					}
					of += sInner
				}
			})
			// Pass 2: build the gradient. A SECOND row walk replaces re-reading the
			// materialized table; the odometer runs once per row, not once per element.
			forEachReduceRow(x.Shape(), axStride, func(i0, of0, inner, sInner int) {
				of := of0
				for p := 0; p < inner; p++ {
					i := i0 + p
					v := xs[i]
					var d float64
					switch numZeros[of] {
					case 0:
						d = ys[of] / v
					case 1:
						if v == 0 {
							d = prodNz[of]
						}
					}
					ds[i] = gs[of] * d
					of += sInner
				}
			})
			return []*tensor.Tensor{gin}, nil
		}
		if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
			xs, ys, gs, ds := xc.Storage().F32(), yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
			// Pass 1: tally zeros and the non-zero product per group.
			forEachReduceRow(x.Shape(), axStride, func(i0, of0, inner, sInner int) {
				of := of0
				for p := 0; p < inner; p++ {
					i := i0 + p
					if xs[i] == 0 {
						numZeros[of]++
					} else {
						prodNz[of] *= float64(xs[i])
					}
					of += sInner
				}
			})
			// Pass 2: build the gradient. A SECOND row walk replaces re-reading the
			// materialized table; the odometer runs once per row, not once per element.
			forEachReduceRow(x.Shape(), axStride, func(i0, of0, inner, sInner int) {
				of := of0
				for p := 0; p < inner; p++ {
					i := i0 + p
					v := float64(xs[i])
					var d float64
					switch numZeros[of] {
					case 0:
						d = float64(ys[of]) / v
					case 1:
						if v == 0 {
							d = prodNz[of]
						}
					}
					ds[i] = float32(float64(gs[of]) * d)
					of += sInner
				}
			})
			return []*tensor.Tensor{gin}, nil
		}
		for i := 0; i < n; i++ { // generic fallback (exotic dtype)
			idx := tensor.Unravel(i, x.Shape())
			of := mapIdx(idx)
			if v := x.AtF64(idx...); v == 0 {
				numZeros[of]++
			} else {
				prodNz[of] *= v
			}
		}
		for i := 0; i < n; i++ {
			idx := tensor.Unravel(i, x.Shape())
			of := mapIdx(idx)
			oidx := tensor.Unravel(of, outShape)
			v := x.AtF64(idx...)
			var d float64
			switch numZeros[of] {
			case 0:
				d = y.AtF64(oidx...) / v
			case 1:
				if v == 0 {
					d = prodNz[of]
				}
			}
			gin.SetF64(g.AtF64(oidx...)*d, idx...)
		}
		return []*tensor.Tensor{gin}, nil
	}
}

func init() {
	RegisterVJP(backend.OpSum, broadcastVJP(false))
	RegisterVJP(backend.OpMean, broadcastVJP(true))
	RegisterVJP(backend.OpMax, extremumVJP())
	RegisterVJP(backend.OpMin, extremumVJP())
	RegisterVJP(backend.OpProd, prodVJP())
	// argmax: index output, non-differentiable → nil grad for its input
	RegisterVJP(backend.OpArgMax, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
		return make([]*tensor.Tensor, len(in)), nil
	})
}

// broadcastFillRows is broadcastVJP's per-element odometer STRIP-MINED over the
// innermost axis (PS4005). The odometer it replaces ran a nested loop with two memory
// operands per ELEMENT purely to advance the source offset, which blocked
// bounds-check elimination on the gather and created a loop-carried dependency that
// killed unrolling — about 1.85 ns of pure index bookkeeping per element.
//
// The structural fact it ignored: axStride[nd-1] is CONSTANT across the innermost
// axis. For Axes {1} on a [512,512] input the stride vector is [1,0], so the entire
// run of 512 elements is a constant fill and the odometer recomputed `of += 0` five
// hundred and twelve times per row. Here the run is handled as one slice — a constant
// fill when the stride is zero, a strided gather otherwise — and the odometer advances
// once per ROW over axes 0..nd-2.
//
// BIT-IDENTICAL UNCONDITIONALLY: this loop accumulates nothing. Every ds[i] is a
// single independent store of gs[of]*scale, so restructuring changes only the order in
// which DISTINCT destinations are written, never a value. The caller still performs
// its one float32(scale) conversion before calling, so the F32 path rounds identically.
//
// Returns false for shapes it does not handle (rank 0, or an innermost extent that
// does not divide the element count), leaving the caller's original loop in place.
func broadcastFillRows[T float32 | float64](ds, gs []T, xs, axStride []int, n int, scale T) bool {
	nd := len(xs)
	if nd == 0 {
		return false
	}
	inner := xs[nd-1]
	if inner <= 0 || n%inner != 0 || len(ds) < n {
		return false
	}
	s := axStride[nd-1]
	coord := make([]int, nd)
	of := 0
	for r := 0; r < n/inner; r++ {
		row := ds[r*inner : r*inner+inner] // re-slice: one bounds check for the run
		if s == 0 {
			v := gs[of] * scale // loop-invariant across the whole run
			for j := range row {
				row[j] = v
			}
		} else {
			for j := range row {
				row[j] = gs[of+j*s] * scale
			}
		}
		// Advance over the OUTER axes only. The innermost axis contributed
		// s*inner on the way up and -s*inner on its wrap, i.e. nothing, so `of`
		// is already the next row's base.
		for ax := nd - 2; ax >= 0; ax-- {
			coord[ax]++
			of += axStride[ax]
			if coord[ax] < xs[ax] {
				break
			}
			coord[ax] = 0
			of -= axStride[ax] * xs[ax]
		}
	}
	return true
}

// forEachReduceRow walks the input in row-major order and hands the caller ONE ROW at a
// time: the flat input index the row starts at, the output offset it maps to, the run
// length, and the output stride within the run.
//
// It replaces reduceOffsets for single-pass consumers. That function materialized the
// whole offset table — an []int of x.Numel() entries, 2 MB of the 3.1 MB a [512,512]
// f32 extremum backward allocated — built by a per-element odometer (PS4005). The
// offsets are now produced as the consumer walks, so the table is gone and the odometer
// ticks once per ROW.
//
// The callback fires once per row, not per element, so the indirect call is amortized
// over the run and each caller keeps its own tight typed inner loop.
//
// TRAVERSAL ORDER IS UNCHANGED — rows ascending, elements ascending within a row — and
// that is load-bearing, not incidental: the extremum VJP routes the gradient to the
// FIRST element attaining each group's maximum, so any reordering would silently move
// which element receives it.
func forEachReduceRow(xShape tensor.Shape, axStride []int, fn func(i0, of, inner, sInner int)) {
	n := xShape.Numel()
	nd := len(xShape)
	if nd == 0 || n == 0 {
		return
	}
	inner, sInner := xShape[nd-1], axStride[nd-1]
	if inner <= 0 || n%inner != 0 {
		inner, sInner = 1, 0 // degenerate shape: one row per element, contract preserved
	}
	coord := make([]int, nd)
	of := 0
	for i0 := 0; i0 < n; i0 += inner {
		fn(i0, of, inner, sInner)
		for ax := nd - 2; ax >= 0; ax-- {
			coord[ax]++
			of += axStride[ax]
			if coord[ax] < xShape[ax] {
				break
			}
			coord[ax] = 0
			of -= axStride[ax] * xShape[ax]
		}
	}
}
