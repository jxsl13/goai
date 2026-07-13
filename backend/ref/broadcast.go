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
