package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BitLinear is the BitNet b1.58 quantization-aware-training linear layer (Ma, Wang,
// Ma, Wang, Wang, Huang, Dong, Wang, Xue & Wei 2024, "The Era of 1-bit LLMs: All
// Large Language Models are in 1.58 Bits", arXiv:2402.17764; architecture from Wang
// et al. 2023, "BitNet: Scaling 1-bit Transformers for Large Language Models",
// arXiv:2310.11453). It trains with TERNARY {−1, 0, +1} weights and 8-bit
// activations: every forward pass fake-quantizes the FULL-PRECISION latent ("master")
// weight W with the paper's absmean quantizer,
//
//	γ  = mean(|W|)                       (scalar over the whole matrix, detached)
//	W̃  = clamp(round(W/γ), −1, +1)       ∈ {−1, 0, +1}
//	Wq = γ·W̃                             (the effective weight used by the matmul)
//
// and the (pre-normalized) input x with the per-token absmax 8-bit quantizer
//
//	s  = 127 / max(|x_row|)              (one scale per row/token, detached)
//	x̃  = clamp(round(x·s), −128, 127)/s ,
//
// both through the straight-through estimator (STE): the composition is
// v + StopGradient(quant(v) − v), so the FORWARD value equals the quantized tensor
// while the BACKWARD gradient passes through to v as identity (exactly the
// round-with-STE idiom of LSQQuantize and FSQ here, and the
// `w + (quant(w) − w).detach()` of the official BitNet code). The optimizer therefore
// updates the full-precision master weights (returned by Params), which drift across
// ternary decision boundaries as training progresses — quantization-aware training
// from scratch, not a post-hoc conversion.
//
// Following the paper, a normalization layer (its SubLN slot; RMSNorm here, as in
// LLaMA and the paper's reference implementation) runs before activation
// quantization.
//
// Contrast with this package's POST-training quantizers — AWQ (AWQScaleSearch),
// GPTQ (GPTQQuantize) and HQQ (HQQQuantize) quantize an ALREADY-TRAINED
// full-precision model, trading accuracy for size after the fact — BitNet instead
// TRAINS the network under ternary constraints so it adapts to them. Contrast with
// LSQ (LSQQuantize), which is also QAT but learns a continuous step size for a
// multi-bit grid by gradient descent; BitNet b1.58 uses a fixed ternary grid whose
// scale γ is recomputed analytically from the weights each forward, never learned.
//
// Storage: a ternary weight carries log2(3) ≈ 1.58 bits of information
// (BitNetBitsPerWeight) versus 32/16 bits for float weights — a >20× storage/memory
// reduction at INFERENCE time (and matmuls that need only additions, since every
// effective weight is ±γ or 0). During training the masters stay full-precision;
// the win is realized when the trained model is exported ternary.
type BitLinear struct {
	// W is the full-precision latent ("master") weight [in, out]. The optimizer
	// updates W; the forward pass uses its ternary fake-quantization γ·W̃.
	W *tensor.Tensor
	// B is the bias [out]; nil (WithBitLinearBias(false)) disables it.
	B *tensor.Tensor
	// Norm is the pre-quantization normalization (the paper's SubLN slot,
	// RMSNorm here); nil (WithBitLinearNorm(false)) disables it.
	Norm   *RMSNorm
	quantW bool // ternary weight quantization on (default true)
	quantA bool // 8-bit activation quantization on (default true)
}

// BitLinearOption configures a BitLinear at construction.
type BitLinearOption func(*BitLinear)

// WithBitLinearBias enables or disables the additive bias (default true).
func WithBitLinearBias(enabled bool) BitLinearOption {
	return func(l *BitLinear) {
		if !enabled {
			l.B = nil
		}
	}
}

// WithBitLinearNorm enables or disables the pre-quantization RMSNorm (default
// true, per the paper's norm-before-quantization).
func WithBitLinearNorm(enabled bool) BitLinearOption {
	return func(l *BitLinear) {
		if !enabled {
			l.Norm = nil
		}
	}
}

// WithBitLinear8BitAct enables or disables the 8-bit activation quantizer
// (default true, the b1.58 configuration).
func WithBitLinear8BitAct(enabled bool) BitLinearOption {
	return func(l *BitLinear) { l.quantA = enabled }
}

// WithBitLinearWeightQuant enables or disables the ternary weight quantizer
// (default true). With both quantizers and the norm disabled, BitLinear reduces
// exactly to a plain Linear on the same weights — useful as a full-precision
// baseline and for wiring tests.
func WithBitLinearWeightQuant(enabled bool) BitLinearOption {
	return func(l *BitLinear) { l.quantW = enabled }
}

// NewBitLinear builds a BitLinear layer with Xavier-uniform full-precision master
// W [in, out], zero bias and a fresh RMSNorm(in), mirroring NewLinear.
// Deterministic via seed.
func NewBitLinear(dtype tensor.Dtype, in, out int, seed uint64, opts ...BitLinearOption) *BitLinear {
	w := tensor.New(dtype, tensor.Shape{in, out})
	XavierUniform(w, in, out, seed)
	b := tensor.New(dtype, tensor.Shape{out})
	Zeros(b)
	l := &BitLinear{W: w, B: b, Norm: NewRMSNorm(dtype, in), quantW: true, quantA: true}
	for _, o := range opts {
		o(l)
	}
	return l
}

// Forward computes norm → 8-bit activation fake-quant → matmul with the ternary
// effective weight γ·W̃ (+ bias if present) for x [batch, in], each stage subject
// to its option. Gradients reach the full-precision masters via the STE.
func (l *BitLinear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: BitLinear expects x rank-2 [batch,in], got %v", x.Shape())
	}
	if x.Shape()[1] != l.W.Shape()[0] {
		return nil, fmt.Errorf("nn: BitLinear in-dim %d != x cols %d", l.W.Shape()[0], x.Shape()[1])
	}
	h := x
	var err error
	if l.Norm != nil {
		if h, err = l.Norm.Forward(ctx, h); err != nil {
			return nil, err
		}
	}
	if l.quantA {
		if h, err = BitQuantizeAct8(ctx, h); err != nil {
			return nil, err
		}
	}
	weff := l.W
	if l.quantW {
		if weff, _, err = BitQuantizeWeight(ctx, l.W); err != nil {
			return nil, err
		}
	}
	y, err := bitExec(ctx, backend.OpMatMul, nil, h, weff)
	if err != nil {
		return nil, err
	}
	if l.B != nil {
		return bitExec(ctx, backend.OpAddBias, nil, y, l.B)
	}
	return y, nil
}

// Params returns the trainable FULL-PRECISION tensors — the latent master weight
// W, the bias (if present) and the norm gain (if present). These are what the
// optimizer updates; the ternary weights exist only inside the forward pass.
func (l *BitLinear) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{l.W}
	if l.B != nil {
		ps = append(ps, l.B)
	}
	if l.Norm != nil {
		ps = append(ps, l.Norm.Gamma)
	}
	return ps
}

// bitnetScaleFloor guards the quantizer scales against division by ~zero (the
// official BitNet code's clamp(min=1e-5)).
const bitnetScaleFloor = 1e-5

// BitQuantizeWeight fake-quantizes w to the BitNet b1.58 ternary grid via the
// absmean quantizer: γ = mean(|w|) (detached, returned), effective weight
// γ·clamp(round(w/γ), −1, +1) ∈ {−γ, 0, +γ}. The round/clamp is composed with
// backend.OpStopGradient as w + sg(quant(w) − w), so the returned tensor equals the
// quantized weight in the forward pass while ∂effective/∂w = 1 (straight-through) —
// gradients reach the full-precision master unchanged.
func BitQuantizeWeight(ctx *backend.Context, w *tensor.Tensor) (*tensor.Tensor, float64, error) {
	n := w.Numel()
	if n == 0 {
		return nil, 0, fmt.Errorf("nn: BitQuantizeWeight got an empty weight tensor")
	}
	var sum float64
	for i := range n {
		sum += math.Abs(w.AtF64(tensor.Unravel(i, w.Shape())...))
	}
	gamma := sum / float64(n)
	scale := math.Max(gamma, bitnetScaleFloor)
	q := tensor.New(w.Dtype(), w.Shape())
	for i := range n {
		c := tensor.Unravel(i, w.Shape())
		t := math.Round(w.AtF64(c...) / scale)
		q.SetF64(scale*math.Max(-1, math.Min(1, t)), c...)
	}
	weff, err := bitSTE(ctx, w, q)
	if err != nil {
		return nil, 0, err
	}
	return weff, gamma, nil
}

// BitQuantizeAct8 fake-quantizes x [batch, d] to 8 bits with the BitNet b1.58
// per-token absmax quantizer: per row, s = 127/max(|x_row|) (detached), quantized
// clamp(round(x·s), −128, 127)/s. Composed with backend.OpStopGradient as
// x + sg(quant(x) − x): forward equals the quantized activations, backward is the
// identity straight-through to x.
func BitQuantizeAct8(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: BitQuantizeAct8 wants rank-2 x[batch,d], got %v", x.Shape())
	}
	rows, cols := x.Shape()[0], x.Shape()[1]
	q := tensor.New(x.Dtype(), x.Shape())
	for r := range rows {
		absmax := 0.0
		for c := range cols {
			if a := math.Abs(x.AtF64(r, c)); a > absmax {
				absmax = a
			}
		}
		s := 127 / math.Max(absmax, bitnetScaleFloor)
		for c := range cols {
			v := math.Max(-128, math.Min(127, math.Round(x.AtF64(r, c)*s)))
			q.SetF64(v/s, r, c)
		}
	}
	return bitSTE(ctx, x, q)
}

// BitNetBitsPerWeight returns the information content of one ternary weight,
// log2(3) ≈ 1.58 bits — the "1.58" of BitNet b1.58. Against 32-bit floats this is
// a >20× storage reduction at inference/export time; during training the master
// weights stay full-precision (the QAT trade: train big, deploy tiny).
func BitNetBitsPerWeight() float64 { return math.Log2(3) }

// bitSTE composes the straight-through estimator v + sg(q − v): forward value q,
// backward identity into v (OpStopGradient has a zero-grad VJP), the same idiom as
// LSQQuantize and FSQ.Quantize.
func bitSTE(ctx *backend.Context, v, q *tensor.Tensor) (*tensor.Tensor, error) {
	delta, err := bitExec(ctx, backend.OpSub, nil, q, v)
	if err != nil {
		return nil, err
	}
	deltaSG, err := bitExec(ctx, backend.OpStopGradient, nil, delta)
	if err != nil {
		return nil, err
	}
	return bitExec(ctx, backend.OpAdd, nil, v, deltaSG)
}

func bitExec(ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, in, at)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
