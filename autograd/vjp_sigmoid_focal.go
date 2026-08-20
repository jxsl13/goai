package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func init() {
	RegisterVJP(backend.OpSigmoidFocalCore, func(ctx *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpSigmoidFocalCoreBackward, []*tensor.Tensor{in[0], in[1], g}, attrs)
		if err != nil {
			return nil, err
		}
		// Labels were folded into detached constants by the original composite
		// graph; preserving that contract means no target gradient, not a trainable
		// zero-valued edge.
		return []*tensor.Tensor{out[0], nil}, nil
	})
}
