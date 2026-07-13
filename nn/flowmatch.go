package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Flow Matching (Lipman, Chen, Ben-Hamu, Nickel & Le 2022, "Flow Matching for
// Generative Modeling", arXiv:2210.02747; Liu, Gong & Liu 2022, "Rectified Flow",
// arXiv:2209.03003; the objective used by Stable Diffusion 3). A generative model is
// trained to transport a simple source distribution (noise x0 ~ N(0,I)) to the data
// distribution (x1) along the straight-line optimal-transport path, by regressing a
// velocity field — no diffusion SDE, no noise schedule, just a mean-squared error.
//
// FlowInterpolate returns the point on the straight-line (rectified / OT) path
// between noise x0 (at t=0) and data x1 (at t=1):
//
//	x_t = (1−t)·x0 + t·x1
//
// The model v_θ is evaluated at (x_t, t) and trained with FlowMatchingLoss to predict
// the path's constant velocity x1 − x0. x0 and x1 are same-shape tensors; t ∈ [0,1].
func FlowInterpolate(ctx *backend.Context, x0, x1 *tensor.Tensor, t float64) (*tensor.Tensor, error) {
	if !x0.Shape().Equal(x1.Shape()) {
		return nil, fmt.Errorf("nn: FlowInterpolate shape mismatch: %v vs %v", x0.Shape(), x1.Shape())
	}
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, nil)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	a, err := ex(backend.OpMul, x0, scalarTensor(x0.Dtype(), 1-t))
	if err != nil {
		return nil, err
	}
	b, err := ex(backend.OpMul, x1, scalarTensor(x1.Dtype(), t))
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, a, b)
}

// FlowMatchingLoss is the Conditional Flow Matching regression loss for the
// straight-line path: the network's predicted velocity vPred (its output at x_t, t)
// is regressed onto the CONSTANT target velocity x1 − x0 (the path's time-derivative,
// independent of t):
//
//	L = mean ‖ vPred − (x1 − x0) ‖²
//
// vPred, x0, x1 are same-shape tensors; the loss is differentiable w.r.t. vPred (the
// model output). x0, x1 are the sampled noise/data.
func FlowMatchingLoss(ctx *backend.Context, vPred, x0, x1 *tensor.Tensor) (*tensor.Tensor, error) {
	if !vPred.Shape().Equal(x0.Shape()) || !x0.Shape().Equal(x1.Shape()) {
		return nil, fmt.Errorf("nn: FlowMatchingLoss shape mismatch: v %v, x0 %v, x1 %v", vPred.Shape(), x0.Shape(), x1.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	target, err := ex(backend.OpSub, nil, x1, x0) // velocity target x1 − x0
	if err != nil {
		return nil, err
	}
	diff, err := ex(backend.OpSub, nil, vPred, target)
	if err != nil {
		return nil, err
	}
	sq, err := ex(backend.OpMul, nil, diff, diff)
	if err != nil {
		return nil, err
	}
	axes := make([]int, sq.Ndim())
	for i := range axes {
		axes[i] = i
	}
	return ex(backend.OpMean, backend.ReduceAttrs{Axes: axes}, sq)
}

// FlowEulerStep integrates one forward Euler step of the learned ODE dx/dt = v:
// x ← x + dt·v. Chaining these from x0 ~ N(0,I) at t=0 to t=1 (with v the model's
// velocity at each step) generates a data sample. For the exact straight-line path a
// single step with dt=1 and the true velocity x1−x0 recovers x1 exactly.
func FlowEulerStep(ctx *backend.Context, x, v *tensor.Tensor, dt float64) (*tensor.Tensor, error) {
	if !x.Shape().Equal(v.Shape()) {
		return nil, fmt.Errorf("nn: FlowEulerStep shape mismatch: x %v, v %v", x.Shape(), v.Shape())
	}
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, nil)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	step, err := ex(backend.OpMul, v, scalarTensor(v.Dtype(), dt))
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, x, step)
}
