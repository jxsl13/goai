package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// transposeKernel returns the transpose of a rank-2 matrix, out[j,i] = in[i,j]
// (numpy.T). Unlike the tensor's view-based Transpose, this is a dispatched op so it
// records on the autograd tape (its gradient is transpose(g)).
func transposeKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: transpose wants 1 input, got %d", len(in))
	}
	x := in[0]
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("ref: transpose needs a rank-2 matrix, got shape %v", x.Shape())
	}
	m, n := x.Shape()[0], x.Shape()[1]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, m})
	for i := range m {
		for j := range n {
			out.SetF64(x.AtF64(i, j), j, i)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpTranspose, tensor.F32, transposeKernel)
	std.add(backend.OpTranspose, tensor.F64, transposeKernel)
}
