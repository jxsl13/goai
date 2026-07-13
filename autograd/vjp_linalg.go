package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Linear-algebra VJPs (§T14). MatMul composes backend ops (transpose views cost
// nothing; the optimized cpu GEMM does the work). dot/nrm2/axpy are small scalar
// loops.

func init() {
	// C = A·B → gA = g·Bᵀ, gB = Aᵀ·g
	RegisterVJP(backend.OpMatMul, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a, b := in[0], in[1]
		bt, err := b.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		ga, err := exec1(ctx, backend.OpMatMul, g, bt)
		if err != nil {
			return nil, err
		}
		at, err := a.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		gb, err := exec1(ctx, backend.OpMatMul, at, g)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{ga, gb}, nil
	})

	// s = Σaᵢbᵢ → (g·b, g·a) with scalar g
	RegisterVJP(backend.OpDot, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a, b := in[0], in[1]
		gv := g.AtF64()
		ga := tensor.New(a.Dtype(), a.Shape())
		gb := tensor.New(b.Dtype(), b.Shape())
		for i := range a.Numel() {
			idx := tensor.Unravel(i, a.Shape())
			ga.SetF64(gv*b.AtF64(idx...), idx...)
			gb.SetF64(gv*a.AtF64(idx...), idx...)
		}
		return []*tensor.Tensor{ga, gb}, nil
	})

	// n = ‖x‖ → g·x/n ; subgradient 0 at x = 0
	RegisterVJP(backend.OpNrm2, func(_ *backend.Context, in, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		norm := out[0].AtF64()
		gv := g.AtF64()
		gin := tensor.New(x.Dtype(), x.Shape())
		if norm != 0 {
			for i := range x.Numel() {
				idx := tensor.Unravel(i, x.Shape())
				gin.SetF64(gv*x.AtF64(idx...)/norm, idx...)
			}
		}
		return []*tensor.Tensor{gin}, nil
	})

	// y = x + b (row-broadcast) → (g, Σᵢ g[i,·])
	// x-grad is g (identity); bias-grad is the column-sum Σ_rows g. Dispatch
	// OpAddBiasBackward on the tape's active backend (like the norm/CE VJPs) — the
	// CPU scalar row-sum was a training-step cost at the FFN's 256×2048 (§T354).
	RegisterVJP(backend.OpAddBias, func(ctx *backend.Context, _, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpAddBiasBackward, []*tensor.Tensor{g}, nil)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{g, out[0]}, nil
	})

	// CE: ∂L/∂z = g·(softmax(z) − q')/b (+z-loss term), targets non-diff (ADR-0007). Dispatch
	// OpCrossEntropyBackward on the tape's active backend (like mhaVJP), so the loss gradient
	// that seeds the whole backward pass runs on the GPU when training on Metal/Vulkan (§T333).
	RegisterVJP(backend.OpCrossEntropy, func(ctx *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpCrossEntropyBackward, []*tensor.Tensor{in[0], in[1], g}, attrs)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out[0], nil}, nil // targets non-differentiable
	})

	// z = αx + y → (α·g, g)
	RegisterVJP(backend.OpAXPY, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		pX, _ := attrs.(backend.AXPYAttrs)
		alpha := pX.WithDefaults().Alpha
		x := in[0]
		gx := tensor.New(x.Dtype(), x.Shape())
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			gx.SetF64(alpha*g.AtF64(idx...), idx...)
		}
		return []*tensor.Tensor{gx, g}, nil
	})
}
