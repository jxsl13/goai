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
func reduceOutMap(xShape tensor.Shape, attrs backend.Attrs) (tensor.Shape, func(idx []int) int, error) {
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
				return nil, nil, fmt.Errorf("autograd: reduce axis %d out of range for rank %d", a, nd)
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
	return outShape, mapIdx, nil
}

// broadcastVJP builds sum/mean gradients: gin[idx] = g[map(idx)] · scale(count).
func broadcastVJP(mean bool) VJP {
	return func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		outShape, mapIdx, err := reduceOutMap(x.Shape(), attrs)
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
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			gv := g.AtF64(tensor.Unravel(mapIdx(idx), outShape)...)
			gin.SetF64(gv*scale, idx...)
		}
		return []*tensor.Tensor{gin}, nil
	}
}

// extremumVJP routes g to the first element attaining the group extremum (§B16).
func extremumVJP() VJP {
	return func(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		outShape, mapIdx, err := reduceOutMap(x.Shape(), attrs)
		if err != nil {
			return nil, err
		}
		gin := tensor.New(x.Dtype(), x.Shape())
		routed := make([]bool, outShape.Numel())
		for i := range x.Numel() { // row-major order → first hit = lowest index
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
		outShape, mapIdx, err := reduceOutMap(x.Shape(), attrs)
		if err != nil {
			return nil, err
		}
		on := outShape.Numel()
		numZeros := make([]int, on)
		prodNz := make([]float64, on) // product of the nonzero elements per group
		for i := range prodNz {
			prodNz[i] = 1
		}
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			of := mapIdx(idx)
			if v := x.AtF64(idx...); v == 0 {
				numZeros[of]++
			} else {
				prodNz[of] *= v
			}
		}
		gin := tensor.New(x.Dtype(), x.Shape())
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			of := mapIdx(idx)
			oidx := tensor.Unravel(of, outShape)
			v := x.AtF64(idx...)
			var d float64 // ∂prod/∂xᵢ = product of the other elements
			switch numZeros[of] {
			case 0:
				d = y.AtF64(oidx...) / v // prodₐₗₗ/xᵢ
			case 1:
				if v == 0 {
					d = prodNz[of] // only the zero element has a nonzero derivative
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
