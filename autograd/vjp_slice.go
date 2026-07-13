package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Slice VJP: the forward extracts x[..,Start:End,..], so the gradient scatters g
// back into a zero tensor of the input's shape at the sliced range (zeros outside
// it) — dx[coords with axis+Start] = g[coords], 0 elsewhere. When Split composes
// several slices of the same x, the tape accumulates their scatters into the full
// gradient.
func init() {
	RegisterVJP(backend.OpSlice, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		pa, _ := attrs.(backend.SliceAttrs)
		ax, _, err := backend.SlicePlan(x.Shape(), pa.Axis, pa.Start, pa.End)
		if err != nil {
			return nil, err
		}
		dx := tensor.New(x.Dtype(), x.Shape()) // zero-initialized
		gShape := g.Shape()
		for e := range g.Numel() {
			gc := tensor.Unravel(e, gShape)
			xc := append([]int(nil), gc...)
			xc[ax] += pa.Start
			dx.SetF64(g.AtF64(gc...), xc...)
		}
		return []*tensor.Tensor{dx}, nil
	})
}
