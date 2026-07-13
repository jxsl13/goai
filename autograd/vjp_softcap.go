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
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			t := y.AtF64(idx...) / pa.Cap // = tanh(x/cap)
			gin.SetF64(g.AtF64(idx...)*(1-t*t), idx...)
		}
		return []*tensor.Tensor{gin}, nil
	})
}
