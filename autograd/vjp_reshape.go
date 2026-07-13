package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Reshape VJP: reshape is a pure re-layout (row-major order preserved), so the
// gradient is g reshaped back to the input's shape — dx[flat i] = g[flat i].
func init() {
	RegisterVJP(backend.OpReshape, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		dx := tensor.New(x.Dtype(), x.Shape())
		xs, gs := x.Shape(), g.Shape()
		for i := range x.Numel() {
			dx.SetF64(g.AtF64(tensor.Unravel(i, gs)...), tensor.Unravel(i, xs)...)
		}
		return []*tensor.Tensor{dx}, nil
	})
}
