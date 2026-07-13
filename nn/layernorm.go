package nn

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LayerNorm normalizes over the last axis (Ba, Kiros & Hinton 2016, "Layer
// Normalization") with learnable gamma/beta, torch semantics (biased variance,
// eps inside the sqrt). Trainable: the VJP for x/gamma/beta lands in autograd
// (§T31), so γ/β receive gradients through the standard dispatch.
type LayerNorm struct {
	Gamma *tensor.Tensor // [d], init 1
	Beta  *tensor.Tensor // [d], init 0
	Eps   float64        // variance-floor epsilon (default 1e-5)
}

// NewLayerNorm builds a LayerNorm over feature size d (γ=1, β=0, eps=1e-5).
func NewLayerNorm(dtype tensor.Dtype, d int) *LayerNorm {
	g := tensor.New(dtype, tensor.Shape{d})
	for i := range d {
		g.SetF64(1, i)
	}
	b := tensor.New(dtype, tensor.Shape{d})
	Zeros(b)
	return &LayerNorm{Gamma: g, Beta: b, Eps: 1e-5}
}

// Forward normalizes x[..., d] through ctx.
func (l *LayerNorm) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpLayerNorm,
		[]*tensor.Tensor{x, l.Gamma, l.Beta}, backend.NormAttrs{Eps: l.Eps})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns gamma and beta.
func (l *LayerNorm) Params() []*tensor.Tensor { return []*tensor.Tensor{l.Gamma, l.Beta} }
