package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DyTDefaultAlphaInit is the default initial value of DyT's learnable scalar α (Zhu et al. 2025,
// §4): 0.5 works for most models (LLMs sometimes tune it higher).
const DyTDefaultAlphaInit = 0.5

// DyT is Dynamic Tanh (Zhu, Chen, He, LeCun & Liu 2025, "Transformers without Normalization",
// CVPR, arXiv:2503.10622) — a drop-in replacement for LayerNorm/RMSNorm that uses NO normalization
// statistics (no mean, no variance, no division). Motivated by the observation that a trained
// LayerNorm maps its inputs through an S-shaped, tanh-like curve, DyT reproduces that mapping
// directly with a learnable elementwise squashing:
//
//	DyT(x) = γ ⊙ tanh(α·x) + β
//
// α is a single learnable SCALAR shared across the whole layer (it controls the effective input
// range before the tanh saturates large activations), while γ (scale) and β (shift) are the same
// per-channel learnable affine vectors as LayerNorm. Because there are no reductions it is cheaper
// than LayerNorm and, the paper shows, matches or beats it across transformers. It is fully
// differentiable (α, γ, β all train) via elementwise ops — no fused kernel.
type DyT struct {
	Alpha *tensor.Tensor // [1], learnable scalar α, init DyTDefaultAlphaInit
	Gamma *tensor.Tensor // [d], per-channel scale, init 1
	Beta  *tensor.Tensor // [d], per-channel shift, init 0
}

// NewDyT builds a DyT layer over feature size d, with α=alphaInit (≤0 → DyTDefaultAlphaInit),
// γ=1 and β=0.
func NewDyT(dtype tensor.Dtype, d int, alphaInit float64) (*DyT, error) {
	if d <= 0 {
		return nil, fmt.Errorf("nn: DyT feature size must be positive, got %d", d)
	}
	if alphaInit <= 0 {
		alphaInit = DyTDefaultAlphaInit
	}
	a := tensor.New(dtype, tensor.Shape{1})
	a.SetF64(alphaInit, 0)
	g := tensor.New(dtype, tensor.Shape{d})
	for i := range d {
		g.SetF64(1, i)
	}
	b := tensor.New(dtype, tensor.Shape{d})
	Zeros(b)
	return &DyT{Alpha: a, Gamma: g, Beta: b}, nil
}

// Forward computes γ ⊙ tanh(α·x) + β over x[..., d].
func (l *DyT) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, nil)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	ax, err := ex(backend.OpMul, x, l.Alpha) // α·x (scalar α broadcasts over all elements)
	if err != nil {
		return nil, err
	}
	t, err := ex(backend.OpTanh, ax)
	if err != nil {
		return nil, err
	}
	gt, err := ex(backend.OpMul, t, l.Gamma) // γ ⊙ tanh(α·x) (per-channel)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, gt, l.Beta) // + β
}

// Params returns the trainable α, γ and β.
func (l *DyT) Params() []*tensor.Tensor { return []*tensor.Tensor{l.Alpha, l.Gamma, l.Beta} }
