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
		case tensor.F32:
			gs, ds := gc.Storage().F32(), gin.Storage().F32()
			sc := float32(scale)
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

// reduceOffsets returns ofs[i] = the flat offset into the contiguous reduced output
// for each row-major input index i, computed with an incremental odometer over
// axStride (one pass, no per-element Unravel) — shared by the extremum/prod VJPs so
// they read the output with a plain slice index instead of a double index round-trip
// (§base-perf, C25, §T635).
func reduceOffsets(xShape tensor.Shape, axStride []int) []int {
	n := xShape.Numel()
	nd := len(xShape)
	ofs := make([]int, n)
	coord := make([]int, nd)
	of := 0
	for i := 0; i < n; i++ {
		ofs[i] = of
		for ax := nd - 1; ax >= 0; ax-- {
			coord[ax]++
			of += axStride[ax]
			if coord[ax] < xShape[ax] {
				break
			}
			coord[ax] = 0
			of -= axStride[ax] * xShape[ax]
		}
	}
	return ofs
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
			ofs := reduceOffsets(x.Shape(), axStride)
			for i := 0; i < n; i++ { // row-major → first hit = lowest index (unchanged)
				of := ofs[i]
				if !routed[of] && xs[i] == ys[of] {
					ds[i] = gs[of]
					routed[of] = true
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
		if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
			xs, ys, gs, ds := xc.Storage().F32(), yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
			ofs := reduceOffsets(x.Shape(), axStride)
			for i := 0; i < n; i++ {
				of := ofs[i]
				if !routed[of] && xs[i] == ys[of] {
					ds[i] = gs[of]
					routed[of] = true
				}
			}
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
			ofs := reduceOffsets(x.Shape(), axStride)
			for i := 0; i < n; i++ {
				if of := ofs[i]; xs[i] == 0 {
					numZeros[of]++
				} else {
					prodNz[of] *= xs[i]
				}
			}
			for i := 0; i < n; i++ {
				of, v := ofs[i], xs[i]
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
			}
			return []*tensor.Tensor{gin}, nil
		}
		if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
			xs, ys, gs, ds := xc.Storage().F32(), yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
			ofs := reduceOffsets(x.Shape(), axStride)
			for i := 0; i < n; i++ {
				if of := ofs[i]; xs[i] == 0 {
					numZeros[of]++
				} else {
					prodNz[of] *= float64(xs[i])
				}
			}
			for i := 0; i < n; i++ {
				of, v := ofs[i], float64(xs[i])
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
			}
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
