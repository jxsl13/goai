package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// addBiasKernel is the reference row-broadcast bias add (§T15): out[i,j] =
// x[i,j] + b[j] for x[m,n], b[n]. The dominant NN case; general broadcasting is
// deferred (§B18).
func addBiasKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: addbias wants 2 inputs, got %d", len(in))
	}
	x, b := in[0], in[1]
	if x.Ndim() != 2 || b.Ndim() != 1 {
		return nil, fmt.Errorf("ref: addbias needs x rank-2 and b rank-1, got %dD, %dD", x.Ndim(), b.Ndim())
	}
	if x.Dtype() != b.Dtype() {
		return nil, fmt.Errorf("ref: addbias dtype mismatch %v vs %v", x.Dtype(), b.Dtype())
	}
	m, n := x.Shape()[0], x.Shape()[1]
	if b.Shape()[0] != n {
		return nil, fmt.Errorf("ref: addbias bias len %d != cols %d", b.Shape()[0], n)
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for i := range m {
		for j := range n {
			out.SetF64(x.AtF64(i, j)+b.AtF64(j), i, j)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpAddBias, tensor.F32, addBiasKernel)
	std.add(backend.OpAddBias, tensor.F64, addBiasKernel)
}
