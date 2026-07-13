package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Transpose VJP: for y = xᵀ, the gradient w.r.t. x is gᵀ (transpose is its own adjoint).
func init() {
	RegisterVJP(backend.OpTranspose, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		m, n := x.Shape()[0], x.Shape()[1]
		gin := tensor.New(x.Dtype(), x.Shape())
		for i := range m {
			for j := range n {
				gin.SetF64(g.AtF64(j, i), i, j) // (gᵀ)[i,j] = g[j,i]
			}
		}
		return []*tensor.Tensor{gin}, nil
	})
}
