package nn

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Layer is a differentiable module. Forward executes through ctx — pass a
// recording context (autograd.Tape.Context()) to train, a plain context for
// inference. Params returns the trainable tensors for optimizers (§T17).
type Layer interface {
	Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error)
	Params() []*tensor.Tensor
}
