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
	coord := make([]int, x.Ndim())
	// Devirtualised traversal (§T646 follow-up): flat typed access with the line
	// base offset built once per line and stepped by the axis stride, instead of
	// the per-element coord→AtF64/SetF64 dispatch (which re-dotted the full coord
	// each element). Same line order, same ascending-axis f64 running sum; F32
	// narrows only the STORED prefix values (sum itself stays f64, exactly like
	// the generic loop) — bit-identical.
	if xs, ok := f64Data(x); ok {
		os, flush, _ := outF64(out) // dtype is F32/F64 here (f64Data ok), cannot fail
		strides := tensor.RowMajorStrides(x.Shape())
		step := strides[ax]
		for l := range reduced.Numel() {
			backend.FillLineCoord(coord, l, reduced, ax)
			base := 0
			for d, c := range coord {
				if d != ax {
					base += c * strides[d]
				}
			}
			var sum float64
			off := base
			for range L {
				sum += xs[off]
				os[off] = sum
				off += step
			}
		}
		flush()
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
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
	std.add(backend.OpCumsum, tensor.F32, cumsumKernel)
	std.add(backend.OpCumsum, tensor.F64, cumsumKernel)
}
