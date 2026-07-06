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

func (s *SwiGLU) exec(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes the SwiGLU FFN for x[..., dim].
func (s *SwiGLU) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Shape()[x.Ndim()-1] != s.Wgate.Shape()[0] {
		return nil, fmt.Errorf("nn: SwiGLU in-dim %d != x last %d", s.Wgate.Shape()[0], x.Shape()[x.Ndim()-1])
	}
	gate, err := s.exec(ctx, backend.OpMatMul, x, s.Wgate)
	if err != nil {
		return nil, err
	}
	act, err := s.exec(ctx, backend.OpSiLU, gate)
	if err != nil {
		return nil, err
	}
	up, err := s.exec(ctx, backend.OpMatMul, x, s.Wup)
	if err != nil {
		return nil, err
	}
	h, err := s.exec(ctx, backend.OpMul, act, up)
	if err != nil {
		return nil, err
	}
	return s.exec(ctx, backend.OpMatMul, h, s.Wdown)
}

// Params returns the three projection matrices.
func (s *SwiGLU) Params() []*tensor.Tensor { return []*tensor.Tensor{s.Wgate, s.Wup, s.Wdown} }
