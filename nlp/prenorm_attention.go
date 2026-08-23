package nlp

import (
	"errors"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// ForwardPreNorm computes x + MHA(LayerNorm(x)) over a packed batch. Backends
// may implement the complete differentiable boundary as OpPreNormAttention;
// unsupported features, dtypes, and layouts retain the established composite.
func (m *MHA) ForwardPreNorm(ctx *backend.Context, x *tensor.Tensor, norm *nn.LayerNorm, batch int) (*tensor.Tensor, error) {
	if m == nil || x == nil || norm == nil {
		return nil, errors.New("nlp: ForwardPreNorm requires non-nil MHA, input, and LayerNorm")
	}
	if ctx == nil {
		ctx = backend.NewContext()
	}
	if m.preNormFusable(ctx, x, norm, batch) {
		out, err := backend.Execute(ctx, backend.OpPreNormAttention, []*tensor.Tensor{
			x, norm.Gamma, norm.Beta, m.Wq, m.Wk, m.Wv, m.Wo,
		}, backend.PreNormAttentionAttrs{Heads: m.Heads, Batch: batch, Eps: norm.Eps})
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	h, err := norm.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	if batch == 1 {
		h, err = m.Forward(ctx, h)
	} else {
		h, err = m.ForwardBatched(ctx, h, batch)
	}
	if err != nil {
		return nil, err
	}
	return exec2(ctx, backend.OpAdd, nil, x, h)
}

func (m *MHA) preNormFusable(ctx *backend.Context, x *tensor.Tensor, norm *nn.LayerNorm, batch int) bool {
	if ctx.Backend == nil || norm.Gamma == nil || norm.Beta == nil ||
		m.Wq == nil || m.Wk == nil || m.Wv == nil || m.Wo == nil ||
		len(m.LoRA) != 0 || len(m.Bias) != 0 || m.Mask != nil || m.Causal ||
		x.Dtype() != tensor.F32 || x.Ndim() != 2 || x.Shape()[0] == 0 || x.Shape()[1] == 0 ||
		batch <= 0 || x.Shape()[0]%batch != 0 || m.Heads <= 0 || x.Shape()[1]%m.Heads != 0 {
		return false
	}
	dim := x.Shape()[1]
	if norm.Gamma.Ndim() != 1 || norm.Gamma.Shape()[0] != dim || norm.Beta.Ndim() != 1 || norm.Beta.Shape()[0] != dim {
		return false
	}
	for _, t := range []*tensor.Tensor{x, norm.Gamma, norm.Beta, m.Wq, m.Wk, m.Wv, m.Wo} {
		if t.Dtype() != tensor.F32 || !t.IsContiguous() || t.Offset() != 0 {
			return false
		}
	}
	for _, w := range []*tensor.Tensor{m.Wq, m.Wk, m.Wv, m.Wo} {
		if w.Ndim() != 2 || w.Shape()[0] != dim || w.Shape()[1] != dim {
			return false
		}
	}
	_, forward := ctx.Backend.Kernel(backend.OpPreNormAttention, tensor.F32)
	_, backward := ctx.Backend.Kernel(backend.OpPreNormAttentionBackward, tensor.F32)
	return forward && backward
}
