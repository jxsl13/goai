package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SoftCap VJP: for y = cap·tanh(x/cap), dy/dx = 1 − tanh²(x/cap) = 1 − (y/cap)², so
// the input cotangent is g·(1 − (y/cap)²) — computed from the forward output y (no
// recompute). Gemma-2 soft-capping is a forward-pass, differentiable operation.
func init() {
	RegisterVJP(backend.OpSoftCap, func(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		pa, _ := attrs.(backend.SoftCapAttrs)
		x, y := in[0], out[0]
		gin := tensor.New(x.Dtype(), x.Shape())
		n := x.Numel()
		cap := pa.Cap
		// t = y/cap is a divide by the loop-invariant cap on EVERY element; hoist it to a
		// reciprocal-multiply (invCap = 1/cap, t = y·invCap). The VJP is a continuous
		// gradient — the ½-ulp reassociation rides the gradient-check tolerance — and ALL
		// THREE paths use the same invCap, so the typed fast paths stay bit-identical to
		// the generic one (the raw-slice-vs-AtF64 parity the tests assert).
		invCap := 1 / cap
		yc, gc := y.Contiguous(), g.Contiguous()
		if x.Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
			ys, gs, ds := yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
			for i := 0; i < n; i++ {
				t := ys[i] * invCap
				ds[i] = gs[i] * (1 - t*t)
			}
			return []*tensor.Tensor{gin}, nil
		}
		if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
			ys, gs, ds := yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
			for i := 0; i < n; i++ {
				t := float64(ys[i]) * invCap
				ds[i] = float32(float64(gs[i]) * (1 - t*t))
			}
			return []*tensor.Tensor{gin}, nil
		}
		for i := range n { // generic per-element fallback
			idx := tensor.Unravel(i, x.Shape())
			t := y.AtF64(idx...) * invCap
			gin.SetF64(g.AtF64(idx...)*(1-t*t), idx...)
		}
		return []*tensor.Tensor{gin}, nil
	})
}
