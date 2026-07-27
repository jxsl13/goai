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
// broadcastRuns copies src into dst over the output shape, hoisting the
// INNERMOST axis out of the odometer. That axis is a single contiguous run in
// dst, and its source stride is 0 (a broadcast axis — one value repeated) or 1
// (a verbatim row), so the run is a fill or a copy; the odometer then ticks once
// per run instead of once per element. Element order and the value written at
// every position are unchanged, so the result is bit-identical to the
// per-element form — the op does no arithmetic at all, it is a same-dtype copy.
func broadcastRuns[T refFloat](dst, src []T, shape, eff []int, n int) {
	ndo := len(shape)
	if ndo == 0 { // rank-0 output: a single element, no axis to hoist
		if n > 0 {
			dst[0] = src[0]
		}
		return
	}
	inner := shape[ndo-1]
	sInner := eff[ndo-1]
	idx := make([]int, ndo)
	ioff, pos := 0, 0
	for pos < n {
		row := dst[pos : pos+inner]
		switch sInner {
		case 0: // broadcast axis: one source value repeated across the run
			v := src[ioff]
			for j := range row {
				row[j] = v
			}
		case 1: // verbatim contiguous row
			copy(row, src[ioff:ioff+inner])
		default: // strided source run (not reachable for row-major inputs; kept total)
			o := ioff
			for j := range row {
				row[j] = src[o]
				o += sInner
			}
		}
		pos += inner
		// The innermost axis completed, so its net contribution to ioff is zero
		// (it advanced by sInner*inner then wrapped by the same amount). Tick the
		// remaining axes exactly as the per-element odometer would.
		for d := ndo - 2; d >= 0; d-- {
			idx[d]++
			ioff += eff[d]
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
			ioff -= eff[d] * shape[d]
		}
	}
}

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
		n := out.Numel()
		switch x.Dtype() {
		case tensor.F64:
			broadcastRuns(out.Storage().F64(), xc.Storage().F64(), pa.Shape, eff, n)
		case tensor.F32:
			broadcastRuns(out.Storage().F32(), xc.Storage().F32(), pa.Shape, eff, n)
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
