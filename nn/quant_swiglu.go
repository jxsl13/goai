package nn

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// QuantSwiGLU is a SwiGLU feed-forward network whose three projections are kept quantized
// (QuantLinear) and dequantized in-kernel — the quantized twin of SwiGLU (§T149). Forward is
// the same computation, SiLU(x·Wgate) ⊙ (x·Wup) then ·Wdown, but each matmul runs on the active
// accelerator (or the CPU fallback) without materializing an f32 weight matrix.
type QuantSwiGLU struct {
	Gate *QuantLinear // In=dim, Out=hidden
	Up   *QuantLinear // In=dim, Out=hidden
	Down *QuantLinear // In=hidden, Out=dim
}

// Forward computes SwiGLU(x): SiLU(x·gate) ⊙ (x·up), projected down. x is [seq, dim] (f32 for
// the accelerator path).
func (s *QuantSwiGLU) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	gate, err := s.Gate.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 quant-matmul-dominated FFN; SiLU dispatch not loop
	act, err := backend.Execute(ctx, backend.OpSiLU, []*tensor.Tensor{gate}, nil)
	if err != nil {
		return nil, err
	}
	up, err := s.Up.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 quant-matmul-dominated FFN; Mul dispatch not loop
	h, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{act[0], up}, nil)
	if err != nil {
		return nil, err
	}
	return s.Down.Forward(ctx, h[0])
}

// Close frees any device-resident weight buffers held by the three projections. Idempotent.
func (s *QuantSwiGLU) Close() error {
	var first error
	for _, l := range []*QuantLinear{s.Gate, s.Up, s.Down} {
		if l != nil {
			if err := l.Close(); err != nil && first == nil {
				first = err
			}
		}
	}
	return first
}
