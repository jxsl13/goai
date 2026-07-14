package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Linear-algebra VJPs (§T14). MatMul composes backend ops (transpose views cost
// nothing; the optimized cpu GEMM does the work). dot/nrm2/axpy are small scalar
// loops.

// scaleInto writes dst[i] = src[i]·scale over contiguous typed storage (dtype-switch
// once, no per-element Unravel/AtF64/SetF64), with a generic fallback for exotic
// dtypes. dst is a fresh contiguous tensor with src's shape (§base-perf, C25).
func scaleInto(dst, src *tensor.Tensor, scale float64) {
	n := src.Numel()
	sc := src.Contiguous()
	switch dst.Dtype() {
	case tensor.F64:
		if sc.Dtype() == tensor.F64 {
			ss, ds := sc.Storage().F64(), dst.Storage().F64()
			for i := 0; i < n; i++ {
				ds[i] = ss[i] * scale
			}
			return
		}
	case tensor.F32:
		if sc.Dtype() == tensor.F32 {
			ss, ds := sc.Storage().F32(), dst.Storage().F32()
			s32 := float32(scale)
			for i := 0; i < n; i++ {
				ds[i] = ss[i] * s32
			}
			return
		}
	}
	for i := 0; i < n; i++ { // generic fallback (mixed dtype / exotic layout)
		idx := tensor.Unravel(i, src.Shape())
		dst.SetF64(src.AtF64(idx...)*scale, idx...)
	}
}

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
		scaleInto(ga, b, gv) // ∂/∂a = g·b
		scaleInto(gb, a, gv) // ∂/∂b = g·a
		return []*tensor.Tensor{ga, gb}, nil
	})

	// n = ‖x‖ → g·x/n ; subgradient 0 at x = 0
	RegisterVJP(backend.OpNrm2, func(_ *backend.Context, in, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		norm := out[0].AtF64()
		gv := g.AtF64()
		gin := tensor.New(x.Dtype(), x.Shape())
		if norm != 0 {
			scaleInto(gin, x, gv/norm) // g·x/‖x‖
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
		scaleInto(gx, g, alpha) // ∂/∂x = α·g
		return []*tensor.Tensor{gx, g}, nil
	})
}
