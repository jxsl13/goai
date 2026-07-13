package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Cumsum VJP: since out[i] = Σ_{j≤i} x[j], each x[j] feeds every out[i] with i≥j, so
// the gradient is the REVERSE cumulative sum — dx[j] = Σ_{i≥j} g[i] along the axis.
func init() {
	RegisterVJP(backend.OpCumsum, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		pa, _ := attrs.(backend.CumsumAttrs)
		ax, reduced, err := backend.CumsumPlan(x.Shape(), pa.Axis)
		if err != nil {
			return nil, err
		}
		L := x.Shape()[ax]
		dx := tensor.New(x.Dtype(), x.Shape())
		coord := make([]int, x.Ndim())
		for l := range reduced.Numel() {
			backend.FillLineCoord(coord, l, reduced, ax)
			var sum float64
			for i := L - 1; i >= 0; i-- { // reverse cumulative sum
				coord[ax] = i
				sum += g.AtF64(coord...)
				dx.SetF64(sum, coord...)
			}
		}
		return []*tensor.Tensor{dx}, nil
	})
}
