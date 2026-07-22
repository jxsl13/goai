package nn

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// RMSNorm is the Llama-style normalization layer (Zhang & Sennrich 2019):
// y = x/√(mean(x²)+eps)·γ over the last axis, no mean subtraction, no bias.
// Trainable via the OpRMSNorm VJP (§T38).
type RMSNorm struct {
	Gamma *tensor.Tensor // [d], init 1
	Eps   float64        // variance-floor epsilon (default 1e-5)
}

// NewRMSNorm builds an RMSNorm over feature size d (γ=1, eps=1e-5).
func NewRMSNorm(dtype tensor.Dtype, d int) *RMSNorm {
	g := tensor.New(dtype, tensor.Shape{d})
	for i := range d {
		g.SetF64(1, i)
	}
	return &RMSNorm{Gamma: g, Eps: 1e-5}
}

// Forward normalizes x[..., d] through ctx.
func (r *RMSNorm) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return execPool2(ctx, backend.OpRMSNorm, backend.NormAttrs{Eps: r.Eps}, x, r.Gamma)
}

// Params returns gamma.
func (r *RMSNorm) Params() []*tensor.Tensor { return []*tensor.Tensor{r.Gamma} }
