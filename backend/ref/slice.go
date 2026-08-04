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
	// Devirtualised block copy (§T646 follow-up): slicing moves one contiguous
	// [outLen·inner] block per outer index, so a same-dtype typed copy() replaces
	// the TWO per-element Unravel allocs + AtF64/SetF64 dispatch (the f64
	// round-trip is exact for a same-dtype copy) — byte-identical.
	if x.Dtype() == tensor.F64 || x.Dtype() == tensor.F32 {
		xs := x.Shape()
		inner := 1
		for d := ax + 1; d < len(xs); d++ {
			inner *= xs[d]
		}
		outer := 1
		for d := 0; d < ax; d++ {
			outer *= xs[d]
		}
		axLen, outLen := xs[ax], outShape[ax]
		n := x.Numel()
		switch x.Dtype() {
		case tensor.F64:
			src := x.Contiguous().Storage().F64()[:n]
			dst := out.Storage().F64()
			for o := 0; o < outer; o++ {
				s := (o*axLen + pa.Start) * inner
				copy(dst[o*outLen*inner:(o+1)*outLen*inner], src[s:s+outLen*inner])
			}
		case tensor.F32:
			src := x.Contiguous().Storage().F32()[:n]
			dst := out.Storage().F32()
			for o := 0; o < outer; o++ {
				s := (o*axLen + pa.Start) * inner
				copy(dst[o*outLen*inner:(o+1)*outLen*inner], src[s:s+outLen*inner])
			}
		}
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for e := range out.Numel() {
		oc := tensor.Unravel(e, outShape)
		xc := append([]int(nil), oc...)
		xc[ax] += pa.Start
		out.SetF64(x.AtF64(xc...), oc...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpSlice, tensor.F32, sliceKernel)
	std.add(backend.OpSlice, tensor.F64, sliceKernel)
}
