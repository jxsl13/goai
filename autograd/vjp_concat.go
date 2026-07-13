package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Concat VJP: the forward joins the inputs along an axis, so the gradient is
// simply g sliced back into each input's segment along that axis — dInₖ[coords] =
// g[coords with axisᵢ += offsetₖ]. One gradient tensor per input.
func init() {
	RegisterVJP(backend.OpConcat, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		pa, _ := attrs.(backend.ConcatAttrs)
		shapes := make([]tensor.Shape, len(in))
		for i, t := range in {
			shapes[i] = t.Shape()
		}
		_, offsets, ax, err := backend.ConcatPlan(shapes, pa.Axis)
		if err != nil {
			return nil, err
		}
		grads := make([]*tensor.Tensor, len(in))
		for i, t := range in {
			off := offsets[i]
			s := t.Shape()
			d := tensor.New(t.Dtype(), s)
			for e := range t.Numel() {
				coords := tensor.Unravel(e, s)
				gc := append([]int(nil), coords...)
				gc[ax] += off
				d.SetF64(g.AtF64(gc...), coords...)
			}
			grads[i] = d
		}
		return grads, nil
	})
}
