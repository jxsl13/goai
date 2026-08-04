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
	// Fused inference path. On a no-recorder (inference) context the three broadcast
	// elementwise ops around the tanh — α·x, γ⊙·, +β — are folded into two typed
	// loops, leaving only OpTanh dispatched. This keeps tanh BIT-IDENTICAL to the
	// dispatch path on every backend (same OpTanh kernel), while the scalar-α prescale
	// and the per-channel γ/β affine are exact (α·x, γ·t, +β), so the whole result is
	// bit-identical to the four-op path — at 1 backend dispatch instead of 4, and
	// without the three full-tensor intermediates. Training (recorder set) keeps the
	// op chain so every parameter still receives gradient.
	if ctx.Recorder == nil {
		d := l.Gamma.Shape()[0]
		if x.Shape()[x.Ndim()-1] == d {
			if fused, ok := l.forwardFused(ctx, x, ex, d); ok {
				return fused, nil
			}
		}
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

// forwardFused implements the fused inference path for F32/F64 (ok=false for other
// dtypes → caller falls back to the op chain). ax = α·x is built in a typed loop,
// tanh'd via the identical OpTanh dispatch, then the per-channel affine γ·t+β is
// applied in a second typed loop that writes the output in place over the tanh
// buffer. Same arithmetic and order as OpMul/OpTanh/OpMul/OpAdd → bit-identical.
//
//perfscan:ignore PS6004 stale line: file is 79 lines, no such code
func (l *DyT) forwardFused(ctx *backend.Context, x *tensor.Tensor, ex func(backend.Op, ...*tensor.Tensor) (*tensor.Tensor, error), d int) (*tensor.Tensor, bool) {
	xc := x.Contiguous()
	ax := tensor.New(x.Dtype(), x.Shape())
	alpha := l.Alpha.AtF64(0)
	switch x.Dtype() {
	case tensor.F64:
		xs, as := xc.Storage().F64(), ax.Storage().F64()
		a := alpha
		for i, v := range xs {
			as[i] = v * a // α·x, matches OpMul(x, α) exactly
		}
	case tensor.F32:
		xs, as := xc.Storage().F32(), ax.Storage().F32()
		a := float32(alpha)
		for i, v := range xs {
			as[i] = v * a
		}
	default:
		return nil, false
	}
	t, err := ex(backend.OpTanh, ax) // identical tanh kernel → bit-exact vs dispatch
	if err != nil {
		return nil, false
	}
	switch x.Dtype() {
	case tensor.F64:
		gs, bs := l.Gamma.Contiguous().Storage().F64(), l.Beta.Contiguous().Storage().F64()
		ts := t.Storage().F64()
		//perfscan:ignore PS3051 stale line: file is 79 lines, Forward is pure dispatch
		for base := 0; base+d <= len(ts); base += d { // row-major: channel = offset within row
			row := ts[base : base+d : base+d]
			for c, tv := range row {
				//perfscan:ignore PS3025,PS5003 stale line: file is 79 lines, no host compute loop | stale line: file is 79 lines, no such code
				row[c] = tv*gs[c] + bs[c] // γ·t + β, matches OpMul(t,γ) then OpAdd(·,β)
			}
		}
	case tensor.F32:
		gs, bs := l.Gamma.Contiguous().Storage().F32(), l.Beta.Contiguous().Storage().F32()
		ts := t.Storage().F32()
		for base := 0; base+d <= len(ts); base += d {
			row := ts[base : base+d : base+d]
			for c, tv := range row {
				//perfscan:ignore PS3025,PS5003 stale line: file is 79 lines, no such code
				row[c] = tv*gs[c] + bs[c]
			}
		}
	}
	return t, true
}

// Params returns the trainable α, γ and β.
func (l *DyT) Params() []*tensor.Tensor { return []*tensor.Tensor{l.Alpha, l.Gamma, l.Beta} }
