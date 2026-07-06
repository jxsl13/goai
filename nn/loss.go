package nn

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Losses (§T16). Both return scalar tensors and are fully differentiable when
// executed through a recording context.

// MSE is the mean squared error mean((pred−target)²), composed from existing
// ops — its gradient comes entirely from the Sub/Mul/Mean VJPs (§T14).
func MSE(ctx *backend.Context, pred, target *tensor.Tensor) (*tensor.Tensor, error) {
	diff, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{pred, target}, nil)
	if err != nil {
		return nil, err
	}
	sq, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
	if err != nil {
		return nil, err
	}
	out, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{sq[0]}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// CrossEntropy is the fused, numerically stable softmax cross-entropy over
// logits[batch,classes] with class-index targets[batch] (ADR-0007). Mean over
// the batch; targets are non-differentiable.
func CrossEntropy(ctx *backend.Context, logits, targets *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpCrossEntropy, []*tensor.Tensor{logits, targets}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
