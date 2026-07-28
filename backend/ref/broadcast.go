package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// broadcastKernel expands x to a larger shape (numpy.broadcast_to): each output
// element reads from the input with broadcast (size-1 or missing) axes mapped to
// index 0. Pure data movement (memory-bound) → reference/cpu only, the GPU backends
// fall back (§I4 / ADR-0008).
func broadcastKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: broadcast wants 1 input, got %d", len(in))
	}
	x := in[0]
	pa, _ := attrs.(backend.BroadcastAttrs)
	offset, err := backend.BroadcastPlan(x.Shape(), pa.Shape)
	if err != nil {
		return nil, err
	}
	xs := x.Shape()
	out := tensor.NewOn(ctx.Device(), x.Dtype(), pa.Shape)
	// Devirtualised traversal (§T646 follow-up): an odometer over the output
	// coordinates carries the input offset incrementally through per-output-axis
	// effective strides (0 for leading/size-1 broadcast axes) instead of the
	// per-element Unravel alloc + coord rebuild + AtF64/SetF64 dispatch. Same
	// row-major element order, same-dtype verbatim copy — bit-identical.
	if x.Dtype() == tensor.F64 || x.Dtype() == tensor.F32 {
		xc := x.Contiguous()
		xStrides := tensor.RowMajorStrides(xs)
		ndo := len(pa.Shape)
		eff := make([]int, ndo)
		for a := range xs {
			if xs[a] != 1 {
				eff[a+offset] = xStrides[a]
			}
		}
		idx := make([]int, ndo)
		n := out.Numel()
		// Peel the innermost output axis out of the odometer: the innermost
		// source axis always maps to output axis ndo-1 with effective stride 1
		// (contiguous, since xc is contiguous) or 0 (that axis is broadcast), so
		// each inner run of length L = pa.Shape[ndo-1] is either a contiguous
		// copy from src (stride 1) or a splat of a single src element (stride 0).
		// The odometer then only carries the outer axes [0, ndo-2], once per run
		// instead of once per element — bit-identical row-major writes.
		switch x.Dtype() {
		case tensor.F64:
			src := xc.Storage().F64()
			dst := out.Storage().F64()
			if ndo == 0 {
				dst[0] = src[0]
				return []*tensor.Tensor{out}, nil
			}
			L := pa.Shape[ndo-1]
			istride := eff[ndo-1] // proven 0 or 1
			ioff := 0
			for pos := 0; pos < n; pos += L {
				run := dst[pos : pos+L]
				if istride == 0 {
					v := src[ioff]
					for k := range run {
						run[k] = v
					}
				} else {
					copy(run, src[ioff:ioff+L])
				}
				for d := ndo - 2; d >= 0; d-- {
					idx[d]++
					ioff += eff[d]
					if idx[d] < pa.Shape[d] {
						break
					}
					idx[d] = 0
					ioff -= eff[d] * pa.Shape[d]
				}
			}
		case tensor.F32:
			src := xc.Storage().F32()
			dst := out.Storage().F32()
			if ndo == 0 {
				dst[0] = src[0]
				return []*tensor.Tensor{out}, nil
			}
			L := pa.Shape[ndo-1]
			istride := eff[ndo-1] // proven 0 or 1
			ioff := 0
			for pos := 0; pos < n; pos += L {
				run := dst[pos : pos+L]
				if istride == 0 {
					v := src[ioff]
					for k := range run {
						run[k] = v
					}
				} else {
					copy(run, src[ioff:ioff+L])
				}
				for d := ndo - 2; d >= 0; d-- {
					idx[d]++
					ioff += eff[d]
					if idx[d] < pa.Shape[d] {
						break
					}
					idx[d] = 0
					ioff -= eff[d] * pa.Shape[d]
				}
			}
		}
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	ic := make([]int, x.Ndim())
	for pos := range out.Numel() {
		oc := tensor.Unravel(pos, pa.Shape)
		for a := range xs {
			if xs[a] == 1 {
				ic[a] = 0
			} else {
				ic[a] = oc[a+offset]
			}
		}
		out.SetF64(x.AtF64(ic...), oc...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpBroadcast, tensor.F32, broadcastKernel)
	std.add(backend.OpBroadcast, tensor.F64, broadcastKernel)
}
