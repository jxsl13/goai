package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// VeRADefaultDInit is the initial value of every entry of VeRA's trainable d vector
// (Kopiczko et al. 2024, §4.1): a small nonzero constant so the down-projection is
// active from the start while the b vector (initialized to zero) keeps ΔW=0.
const VeRADefaultDInit = 0.1

// VeRAMatrices holds the FROZEN random low-rank matrices A and B that VeRA SHARES
// (ties) across every adapted layer (Kopiczko, Blankevoort & Asano 2024, "VeRA:
// Vector-based Random Matrix Adaptation", ICLR 2024, arXiv:2310.11454, §R215). In
// our [in,out] row-vector convention A is [in,r] and B is [r,out]. They are drawn
// once, never trained, and reused by every VeRALinear — that sharing is what makes
// VeRA's trainable footprint an order of magnitude smaller than LoRA's.
type VeRAMatrices struct {
	A *tensor.Tensor // [in, r], frozen shared random
	B *tensor.Tensor // [r, out], frozen shared random
	R int            // rank r
}

// NewVeRAMatrices draws the single shared frozen (A[in,r], B[r,out]) pair from seed
// (Kaiming-uniform, like LoRA's A). Build every layer's adapter over the SAME
// returned value to realise VeRA's parameter savings.
func NewVeRAMatrices(in, out, r int, dtype tensor.Dtype, seed uint64) (*VeRAMatrices, error) {
	if in <= 0 || out <= 0 {
		return nil, fmt.Errorf("nn: VeRA dims must be positive, got in=%d out=%d", in, out)
	}
	if r <= 0 || r > in || r > out {
		return nil, fmt.Errorf("nn: VeRA rank %d out of range for [%d,%d]", r, in, out)
	}
	a := tensor.New(dtype, tensor.Shape{in, r})
	KaimingUniform(a, in, seed)
	b := tensor.New(dtype, tensor.Shape{r, out})
	KaimingUniform(b, r, seed+1)
	return &VeRAMatrices{A: a, B: b, R: r}, nil
}

// VeRALinear is a Vector-based Random Matrix Adaptation adapter over a FROZEN base
// weight (Kopiczko et al. 2024, §R215). For our [in,out] convention, with the
// shared frozen random A[in,r], B[r,out] and the two PER-LAYER trainable scaling
// vectors d[r] and b[out] (Λ_d=diag(d), Λ_b=diag(b) in the paper):
//
//	y = x·W + b ⊙ ( ( (x·A) ⊙ d ) · B )
//
// Only d and b train — A, B and W are frozen — so a layer adds just r+out trainable
// parameters (the vectors), versus LoRA's r·(in+out) matrices (§R27). d starts at a
// small constant (VeRADefaultDInit) and b starts at ZERO, so ΔW=0 at initialization
// and training begins from the exact pretrained behaviour. The two ⊙ rescalings are
// (IA)³-style per-feature scalings (OpIA3), each differentiable w.r.t. its vector.
type VeRALinear struct {
	W  *tensor.Tensor // [in, out], frozen base
	A  *tensor.Tensor // [in, r], frozen shared random
	B  *tensor.Tensor // [r, out], frozen shared random
	D  *tensor.Tensor // [r], trainable, init VeRADefaultDInit
	Bv *tensor.Tensor // [out], trainable, init 0
	R  int            // rank r
}

// NewVeRA wraps a frozen base weight w[in,out] with a VeRA adapter that reuses the
// shared frozen matrices m (whose shapes must match w). d is initialized to dInit
// (≤0 → VeRADefaultDInit) and b to zero.
func NewVeRA(w *tensor.Tensor, m *VeRAMatrices, dInit float64) (*VeRALinear, error) {
	if w.Ndim() != 2 {
		return nil, fmt.Errorf("nn: VeRA base must be [in,out], got %v", w.Shape())
	}
	in, out := w.Shape()[0], w.Shape()[1]
	if !m.A.Shape().Equal(tensor.Shape{in, m.R}) || !m.B.Shape().Equal(tensor.Shape{m.R, out}) {
		return nil, fmt.Errorf("nn: VeRA shared matrices A%v B%v do not fit base [%d,%d] rank %d", m.A.Shape(), m.B.Shape(), in, out, m.R)
	}
	if dInit <= 0 {
		dInit = VeRADefaultDInit
	}
	d := tensor.New(w.Dtype(), tensor.Shape{m.R})
	for i := range m.R {
		d.SetF64(dInit, i)
	}
	bv := tensor.New(w.Dtype(), tensor.Shape{out})
	Zeros(bv) // b = 0 → ΔW starts at 0
	return &VeRALinear{W: w, A: m.A, B: m.B, D: d, Bv: bv, R: m.R}, nil
}

func (v *VeRALinear) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes y = x·W + b ⊙ (((x·A) ⊙ d)·B).
func (v *VeRALinear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	base, err := v.exec(ctx, backend.OpMatMul, nil, x, v.W)
	if err != nil {
		return nil, err
	}
	xa, err := v.exec(ctx, backend.OpMatMul, nil, x, v.A) // [.., r]
	if err != nil {
		return nil, err
	}
	xad, err := v.exec(ctx, backend.OpIA3, nil, xa, v.D) // (x·A) ⊙ d
	if err != nil {
		return nil, err
	}
	xadb, err := v.exec(ctx, backend.OpMatMul, nil, xad, v.B) // [.., out]
	if err != nil {
		return nil, err
	}
	delta, err := v.exec(ctx, backend.OpIA3, nil, xadb, v.Bv) // ⊙ b
	if err != nil {
		return nil, err
	}
	return v.exec(ctx, backend.OpAdd, nil, base, delta)
}

// Params returns only the trainable per-layer scaling vectors d and b (A, B and W
// are frozen and shared) — VeRA's tiny trainable footprint.
func (v *VeRALinear) Params() []*tensor.Tensor { return []*tensor.Tensor{v.D, v.Bv} }
