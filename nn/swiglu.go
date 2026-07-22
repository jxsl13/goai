package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SwiGLU is the Llama feed-forward (Shazeer 2020, §R30):
// FFN(x) = ( SiLU(x·Wgate) ⊙ (x·Wup) ) · Wdown, no bias. Fully differentiable
// (composition of matmul/silu/mul), so gradients reach all three matrices.
type SwiGLU struct {
	Wgate, Wup *tensor.Tensor // [dim, hidden]
	Wdown      *tensor.Tensor // [hidden, dim]
}

// NewSwiGLU builds a SwiGLU FFN with Xavier-uniform weights.
func NewSwiGLU(dtype tensor.Dtype, dim, hidden int, seed uint64) *SwiGLU {
	wg := tensor.New(dtype, tensor.Shape{dim, hidden})
	XavierUniform(wg, dim, hidden, seed)
	wu := tensor.New(dtype, tensor.Shape{dim, hidden})
	XavierUniform(wu, dim, hidden, seed+1)
	wd := tensor.New(dtype, tensor.Shape{hidden, dim})
	XavierUniform(wd, hidden, dim, seed+2)
	return &SwiGLU{Wgate: wg, Wup: wu, Wdown: wd}
}

// Forward computes the SwiGLU FFN for x[..., dim]. The five ops route through the
// recorder-guarded execPool helpers (T964), so an inference FFN reuses one pooled input
// slice per op instead of allocating a fresh one; under a tape (training) they fall back
// to a fresh slice the tape node can keep — see execpool.go.
func (s *SwiGLU) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Shape()[x.Ndim()-1] != s.Wgate.Shape()[0] {
		return nil, fmt.Errorf("nn: SwiGLU in-dim %d != x last %d", s.Wgate.Shape()[0], x.Shape()[x.Ndim()-1])
	}
	gate, err := execPool2(ctx, backend.OpMatMul, nil, x, s.Wgate)
	if err != nil {
		return nil, err
	}
	act, err := execPool1(ctx, backend.OpSiLU, nil, gate)
	if err != nil {
		return nil, err
	}
	up, err := execPool2(ctx, backend.OpMatMul, nil, x, s.Wup)
	if err != nil {
		return nil, err
	}
	h, err := execPool2(ctx, backend.OpMul, nil, act, up)
	if err != nil {
		return nil, err
	}
	return execPool2(ctx, backend.OpMatMul, nil, h, s.Wdown)
}

// Params returns the three projection matrices.
func (s *SwiGLU) Params() []*tensor.Tensor { return []*tensor.Tensor{s.Wgate, s.Wup, s.Wdown} }
