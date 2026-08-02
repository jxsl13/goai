package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Linear is the fully-connected layer y = x·W + b with x[batch,in], W[in,out],
// b[out]. W is stored [in,out] so the forward pass is a plain row-major matmul
// with no transpose (torch stores [out,in] and computes x·Wᵀ — same math,
// different layout; documented divergence). Both ops run through the backend
// dispatch, so gradients flow with zero layer-specific autograd code (§T5/§T13).
type Linear struct {
	W *tensor.Tensor // [in, out]
	B *tensor.Tensor // [out]; nil → no bias
}

// NewLinear builds a Linear layer with Xavier-uniform W (Glorot 2010) and zero
// bias — the standard default for tanh/sigmoid nets; call KaimingNormal on W for
// ReLU stacks (He 2015). Deterministic via seed.
func NewLinear(dtype tensor.Dtype, in, out int, seed uint64) *Linear {
	w := tensor.New(dtype, tensor.Shape{in, out})
	XavierUniform(w, in, out, seed)
	b := tensor.New(dtype, tensor.Shape{out})
	Zeros(b)
	return &Linear{W: w, B: b}
}

// Forward computes x·W (+ b if present) through ctx.
func (l *Linear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: Linear expects x rank-2 [batch,in], got %v", x.Shape())
	}
	if x.Shape()[1] != l.W.Shape()[0] {
		return nil, fmt.Errorf("nn: Linear in-dim %d != x cols %d", l.W.Shape()[0], x.Shape()[1])
	}
	// execPool2 rather than a slice literal per call: this is the most-called forward in the
	// package — every MLP block of every model routes through it — and each literal is one
	// allocation Execute drops the instant it returns. The helper defers to a fresh slice under a
	// recorder, because the tape node stores that exact slice.
	y, err := execPool2(ctx, backend.OpMatMul, nil, x, l.W)
	if err != nil {
		return nil, err
	}
	if l.B != nil {
		if y, err = execPool2(ctx, backend.OpAddBias, nil, y, l.B); err != nil {
			return nil, err
		}
	}
	return y, nil
}

// Params returns the trainable tensors (W, and B if present).
func (l *Linear) Params() []*tensor.Tensor {
	if l.B == nil {
		return []*tensor.Tensor{l.W}
	}
	return []*tensor.Tensor{l.W, l.B}
}
