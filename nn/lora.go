package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LoRALinear is a low-rank adapter over a FROZEN base weight (Hu et al. 2021,
// §R27). For our [in,out] weight convention:
//
//	y = x·W + (alpha/r)·(x·A)·B ,   A[in,r], B[r,out], r ≪ min(in,out)
//
// W is frozen (not in Params); only A and B train. A is Kaiming-initialized and
// B is zero, so the adapter starts as a no-op (ΔW = BA = 0) — training begins
// from the exact pretrained behaviour.
type LoRALinear struct {
	W     *tensor.Tensor // [in, out], frozen base
	A     *tensor.Tensor // [in, r]
	B     *tensor.Tensor // [r, out]
	R     int            // LoRA rank
	Alpha float64        // LoRA scaling α; effective scale α/R
}

// NewLoRA wraps a frozen base weight w[in,out] with a rank-r adapter.
func NewLoRA(w *tensor.Tensor, r int, alpha float64, seed uint64) (*LoRALinear, error) {
	if w.Ndim() != 2 {
		return nil, fmt.Errorf("nn: LoRA base must be [in,out], got %v", w.Shape())
	}
	in, out := w.Shape()[0], w.Shape()[1]
	if r <= 0 || r > in || r > out {
		return nil, fmt.Errorf("nn: LoRA rank %d out of range for [%d,%d]", r, in, out)
	}
	a := tensor.New(w.Dtype(), tensor.Shape{in, r})
	KaimingUniform(a, in, seed) // random A
	b := tensor.New(w.Dtype(), tensor.Shape{r, out})
	Zeros(b) // B = 0 → ΔW starts at 0
	return &LoRALinear{W: w, A: a, B: b, R: r, Alpha: alpha}, nil
}

func (l *LoRALinear) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes y = x·W + (alpha/r)·(x·A)·B.
func (l *LoRALinear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 x·W matmul dispatch, matmul-dominated forward, no Go loop
	base, err := l.exec(ctx, backend.OpMatMul, nil, x, l.W)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 x·A matmul dispatch, matmul-dominated forward
	xa, err := l.exec(ctx, backend.OpMatMul, nil, x, l.A)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 xa·B matmul dispatch, matmul-dominated forward
	delta, err := l.exec(ctx, backend.OpMatMul, nil, xa, l.B)
	if err != nil {
		return nil, err
	}
	// y = base + (alpha/r)·delta  via axpy
	//perfscan:ignore PS3024 AXPY dispatch, memory-bound tail of matmul forward
	return l.exec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: l.Alpha / float64(l.R)}, delta, base)
}

// Params returns only the trainable adapter matrices A and B (W is frozen).
func (l *LoRALinear) Params() []*tensor.Tensor { return []*tensor.Tensor{l.A, l.B} }
