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
	axes := attrs.Ints("axes")
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
		keepdims := attrs.Bool("keepdims", false)
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
	axis := attrs.Int("axis", reduceAllAxis)

	if axis == reduceAllAxis {
		best := math.Inf(-1)
		bi := 0
		for pos := range x.Numel() {
			v := x.AtF64(tensor.Unravel(pos, x.Shape())...)
			if v > best {
				best, bi = v, pos
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
	reg(backend.OpArgMax, argmaxKernel)
}
