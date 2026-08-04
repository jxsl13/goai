package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// cumsumKernel is the cumulative sum along an axis (numpy.cumsum): out[..,i,..] =
// Σ_{j≤i} x[..,j,..] along the axis, in ascending order (so the per-element
// accumulation order is deterministic; f64 accumulation, §V10). Memory-bound → the
// reference/cpu computes it; the GPU backends fall back (§I4 / ADR-0008).
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func cumsumKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: cumsum wants 1 input, got %d", len(in))
	}
	x := in[0]
	pa, _ := attrs.(backend.CumsumAttrs)
	ax, reduced, err := backend.CumsumPlan(x.Shape(), pa.Axis)
	if err != nil {
		return nil, err
	}
	L := x.Shape()[ax]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	// Devirtualised traversal (§T646 follow-up): flat typed access instead of the
	// per-element coord→AtF64/SetF64 dispatch. This uses the same inner/outer flat
	// decomposition the OpCumsum VJP already proves correct — no per-line
	// FillLineCoord/Unravel (which heap-allocated a coord slice on every line) and no
	// per-line base recompute. For a row-major tensor strides[ax] == inner, so each
	// line's flat offsets are base + i·inner for i ascending, base = o·L·inner + j —
	// exactly the offsets the coord-odometer visited, in the same order. F32 narrows
	// only the STORED prefix values (sum stays f64, like the generic loop) —
	// bit-identical.
	if xs, ok := f64Data(x); ok {
		os, flush, _ := outF64(out) // dtype is F32/F64 here (f64Data ok), cannot fail
		inner := 1
		for d := ax + 1; d < x.Ndim(); d++ {
			inner *= x.Shape()[d]
		}
		outer := 1
		for d := 0; d < ax; d++ {
			outer *= x.Shape()[d]
		}
		for o := 0; o < outer; o++ {
			for j := 0; j < inner; j++ {
				var sum float64
				off := o*L*inner + j
				for range L {
					sum += xs[off]
					os[off] = sum
					off += inner
				}
			}
		}
		flush()
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	coord := make([]int, x.Ndim())
	for l := range reduced.Numel() {
		backend.FillLineCoord(coord, l, reduced, ax)
		var sum float64
		for i := range L {
			coord[ax] = i
			sum += x.AtF64(coord...)
			out.SetF64(sum, coord...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpCumsum, tensor.F32, cumsumKernel)
	std.add(backend.OpCumsum, tensor.F64, cumsumKernel)
}
