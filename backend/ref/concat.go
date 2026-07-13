package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// concatKernel joins N tensors along an axis (numpy.concatenate): out has the same
// shape as the inputs except along the concat axis, whose extent is the sum of the
// inputs'. Each input's elements are copied to the output with the axis coordinate
// shifted by that input's running offset. All inputs share a dtype (that of the
// first). Pure data movement — memory-bound, so it stays on the reference/cpu (the
// GPU backends fall back, §I4 / ADR-0008).
func concatKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 0 {
		return nil, fmt.Errorf("ref: concat needs ≥1 input")
	}
	pa, _ := attrs.(backend.ConcatAttrs)
	shapes := make([]tensor.Shape, len(in))
	for i, t := range in {
		shapes[i] = t.Shape()
		if t.Dtype() != in[0].Dtype() {
			return nil, fmt.Errorf("ref: concat dtype mismatch %v vs %v", t.Dtype(), in[0].Dtype())
		}
	}
	outShape, offsets, ax, err := backend.ConcatPlan(shapes, pa.Axis)
	if err != nil {
		return nil, err
	}
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), outShape)
	for i, t := range in {
		off := offsets[i]
		s := t.Shape()
		for e := range t.Numel() {
			coords := tensor.Unravel(e, s)
			oc := append([]int(nil), coords...)
			oc[ax] += off
			out.SetF64(t.AtF64(coords...), oc...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpConcat, tensor.F32, concatKernel)
	std.add(backend.OpConcat, tensor.F64, concatKernel)
}
