package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// sliceKernel extracts the half-open range [Start,End) along an axis (numpy basic
// slicing x[..,Start:End,..]): the output has the input's shape except that axis,
// whose extent becomes End−Start. Each output element copies from the input with
// the axis coordinate shifted by +Start. Pure data movement → reference/cpu only
// (memory-bound, the GPU backends fall back §I4 / ADR-0008).
func sliceKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: slice wants 1 input, got %d", len(in))
	}
	x := in[0]
	pa, _ := attrs.(backend.SliceAttrs)
	ax, outShape, err := backend.SlicePlan(x.Shape(), pa.Axis, pa.Start, pa.End)
	if err != nil {
		return nil, err
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), outShape)
	for e := range out.Numel() {
		oc := tensor.Unravel(e, outShape)
		xc := append([]int(nil), oc...)
		xc[ax] += pa.Start
		out.SetF64(x.AtF64(xc...), oc...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpSlice, tensor.F32, sliceKernel)
	std.add(backend.OpSlice, tensor.F64, sliceKernel)
}
