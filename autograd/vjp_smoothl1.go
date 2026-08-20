package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func init() {
	RegisterVJP(backend.OpSmoothL1Core, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		return backend.Execute(ctx, backend.OpSmoothL1CoreBackward, []*tensor.Tensor{in[0], in[1], g}, nil)
	})
}
