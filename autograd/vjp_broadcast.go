package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Broadcast VJP: broadcasting replicates the input over the broadcast axes, so the
// gradient is the SUM of the output gradient over those axes, back into the input's
// shape — dx[ic] = Σ over output positions mapping to ic of g[oc]. Mirrors the
// forward coordinate mapping (BroadcastPlan) so they cannot disagree.
func init() {
	RegisterVJP(backend.OpBroadcast, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		pa, _ := attrs.(backend.BroadcastAttrs)
		offset, err := backend.BroadcastPlan(x.Shape(), pa.Shape)
		if err != nil {
			return nil, err
		}
		xs := x.Shape()
		dx := tensor.New(x.Dtype(), xs) // zero-initialized accumulator
		gShape := g.Shape()
		// Typed fast paths for contiguous f32/f64 g: one incremental-offset walk
		// (see bcastSumInto), no per-element Unravel/AtF64/SetF64.
		if g.IsContiguous() && g.Numel() > 0 {
			b := g.Offset()
			n := g.Numel()
			switch g.Dtype() {
			case tensor.F64:
				bcastSumInto(dx.Storage().F64(), g.Storage().F64()[b:b+n], gShape, xs, offset)
				return []*tensor.Tensor{dx}, nil
			case tensor.F32:
				bcastSumInto(dx.Storage().F32(), g.Storage().F32()[b:b+n], gShape, xs, offset)
				return []*tensor.Tensor{dx}, nil
			}
		}
		ic := make([]int, x.Ndim())
		for pos := range g.Numel() {
			oc := tensor.Unravel(pos, gShape)
			for a := range xs {
				if xs[a] == 1 {
					ic[a] = 0
				} else {
					ic[a] = oc[a+offset]
				}
			}
			dx.SetF64(dx.AtF64(ic...)+g.AtF64(oc...), ic...)
		}
		return []*tensor.Tensor{dx}, nil
	})
}
