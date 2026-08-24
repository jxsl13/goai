package nlp

import (
	"errors"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// PreNormTransformerBlock groups the layers of one complete pre-normalized
// attention-plus-FFN transformer block for stack execution.
type PreNormTransformerBlock struct {
	// Attention supplies the self-attention projections and head geometry.
	Attention *MHA
	// Norm1 normalizes the attention input.
	Norm1 *nn.LayerNorm
	// Norm2 normalizes the feed-forward input.
	Norm2 *nn.LayerNorm
	// Up expands the feed-forward hidden dimension before exact GELU.
	Up *nn.Linear
	// Down projects the feed-forward activation back to the model dimension.
	Down *nn.Linear
}

// ForwardPreNormTransformerStack computes blocks sequentially. Supported
// backends may keep every intermediate activation inside one differentiable
// operation; all other cases loop over ForwardPreNormTransformerBlock exactly.
func ForwardPreNormTransformerStack(ctx *backend.Context, x *tensor.Tensor,
	blocks []PreNormTransformerBlock, batch int) (*tensor.Tensor, error) {
	if x == nil {
		return nil, errors.New("nlp: ForwardPreNormTransformerStack requires non-nil input")
	}
	if len(blocks) == 0 {
		return x, nil
	}
	if ctx == nil {
		ctx = backend.NewContext()
	}
	if preNormTransformerStackFusable(ctx, x, blocks, batch) {
		inputs := make([]*tensor.Tensor, 1, 1+12*len(blocks))
		inputs[0] = x
		for _, block := range blocks {
			inputs = append(inputs,
				block.Norm1.Gamma, block.Norm1.Beta,
				block.Attention.Wq, block.Attention.Wk, block.Attention.Wv, block.Attention.Wo,
				block.Norm2.Gamma, block.Norm2.Beta,
				block.Up.W, block.Up.B, block.Down.W, block.Down.B,
			)
		}
		first := blocks[0]
		out, err := backend.Execute(ctx, backend.OpPreNormTransformerStack, inputs, backend.PreNormTransformerStackAttrs{
			Depth: len(blocks), Heads: first.Attention.Heads, Batch: batch,
			Eps1: first.Norm1.Eps, Eps2: first.Norm2.Eps,
		})
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	h := x
	var err error
	for _, block := range blocks {
		h, err = ForwardPreNormTransformerBlock(ctx, h, block.Attention, block.Norm1, block.Norm2, block.Up, block.Down, batch)
		if err != nil {
			return nil, err
		}
	}
	return h, nil
}

func preNormTransformerStackFusable(ctx *backend.Context, x *tensor.Tensor,
	blocks []PreNormTransformerBlock, batch int) bool {
	if ctx.Backend == nil || len(blocks) < 2 || len(blocks) > 8 {
		return false
	}
	first := blocks[0]
	if first.Attention == nil || first.Norm1 == nil || first.Norm2 == nil || first.Up == nil || first.Down == nil {
		return false
	}
	heads, hidden := first.Attention.Heads, 0
	if first.Up.W != nil && first.Up.W.Ndim() == 2 {
		hidden = first.Up.W.Shape()[1]
	}
	for _, block := range blocks {
		if block.Attention == nil || block.Norm1 == nil || block.Norm2 == nil || block.Up == nil || block.Down == nil ||
			block.Attention.Heads != heads || block.Norm1.Eps != first.Norm1.Eps || block.Norm2.Eps != first.Norm2.Eps ||
			block.Up.W == nil || block.Up.W.Ndim() != 2 || block.Up.W.Shape()[1] != hidden ||
			!preNormTransformerBlockInputsFusable(x, block.Attention, block.Norm1, block.Norm2, block.Up, block.Down, batch) {
			return false
		}
	}
	_, forward := ctx.Backend.Kernel(backend.OpPreNormTransformerStack, tensor.F32)
	_, backward := ctx.Backend.Kernel(backend.OpPreNormTransformerStackBackward, tensor.F32)
	return forward && backward
}

// ForwardPreNormTransformerBlock computes an attention residual followed by an
// exact-GELU FFN residual. Supported backends may execute the complete
// differentiable block as one operation; all other cases retain the established
// pre-norm attention and FFN helpers.
func ForwardPreNormTransformerBlock(ctx *backend.Context, x *tensor.Tensor, attention *MHA,
	norm1, norm2 *nn.LayerNorm, up, down *nn.Linear, batch int) (*tensor.Tensor, error) {
	if x == nil || attention == nil || norm1 == nil || norm2 == nil || up == nil || down == nil {
		return nil, errors.New("nlp: ForwardPreNormTransformerBlock requires non-nil input and layers")
	}
	if norm1.Gamma == nil || norm1.Beta == nil || norm2.Gamma == nil || norm2.Beta == nil ||
		attention.Wq == nil || attention.Wk == nil || attention.Wv == nil || attention.Wo == nil ||
		up.W == nil || up.B == nil || down.W == nil || down.B == nil {
		return nil, errors.New("nlp: ForwardPreNormTransformerBlock requires complete attention, LayerNorm, and biased Linear parameters")
	}
	if ctx == nil {
		ctx = backend.NewContext()
	}
	if preNormTransformerBlockFusable(ctx, x, attention, norm1, norm2, up, down, batch) {
		out, err := backend.Execute(ctx, backend.OpPreNormTransformerBlock, []*tensor.Tensor{
			x, norm1.Gamma, norm1.Beta,
			attention.Wq, attention.Wk, attention.Wv, attention.Wo,
			norm2.Gamma, norm2.Beta, up.W, up.B, down.W, down.B,
		}, backend.PreNormTransformerBlockAttrs{
			Heads: attention.Heads, Batch: batch, Eps1: norm1.Eps, Eps2: norm2.Eps,
		})
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	h, err := attention.ForwardPreNorm(ctx, x, norm1, batch)
	if err != nil {
		return nil, err
	}
	return nn.ForwardPreNormFFN(ctx, h, norm2, up, down)
}

func preNormTransformerBlockFusable(ctx *backend.Context, x *tensor.Tensor, attention *MHA,
	norm1, norm2 *nn.LayerNorm, up, down *nn.Linear, batch int) bool {
	if ctx.Backend == nil || !preNormTransformerBlockInputsFusable(x, attention, norm1, norm2, up, down, batch) {
		return false
	}
	_, forward := ctx.Backend.Kernel(backend.OpPreNormTransformerBlock, tensor.F32)
	_, backward := ctx.Backend.Kernel(backend.OpPreNormTransformerBlockBackward, tensor.F32)
	return forward && backward
}

func preNormTransformerBlockInputsFusable(x *tensor.Tensor, attention *MHA,
	norm1, norm2 *nn.LayerNorm, up, down *nn.Linear, batch int) bool {
	if len(attention.LoRA) != 0 || len(attention.Bias) != 0 || attention.Mask != nil || attention.Causal ||
		x.Dtype() != tensor.F32 || x.Ndim() != 2 || x.Shape()[0] == 0 || x.Shape()[1] == 0 ||
		batch <= 0 || x.Shape()[0]%batch != 0 || attention.Heads <= 0 || x.Shape()[1]%attention.Heads != 0 {
		return false
	}
	dim := x.Shape()[1]
	if norm1.Gamma.Ndim() != 1 || norm1.Gamma.Shape()[0] != dim || norm1.Beta.Ndim() != 1 || norm1.Beta.Shape()[0] != dim ||
		norm2.Gamma.Ndim() != 1 || norm2.Gamma.Shape()[0] != dim || norm2.Beta.Ndim() != 1 || norm2.Beta.Shape()[0] != dim ||
		up.W.Ndim() != 2 || up.W.Shape()[0] != dim || up.W.Shape()[1] == 0 || up.B.Ndim() != 1 || up.B.Shape()[0] != up.W.Shape()[1] ||
		down.W.Ndim() != 2 || down.W.Shape()[0] != up.W.Shape()[1] || down.W.Shape()[1] != dim || down.B.Ndim() != 1 || down.B.Shape()[0] != dim {
		return false
	}
	for _, weight := range []*tensor.Tensor{attention.Wq, attention.Wk, attention.Wv, attention.Wo} {
		if weight.Ndim() != 2 || weight.Shape()[0] != dim || weight.Shape()[1] != dim {
			return false
		}
	}
	for _, value := range []*tensor.Tensor{
		x, norm1.Gamma, norm1.Beta, attention.Wq, attention.Wk, attention.Wv, attention.Wo,
		norm2.Gamma, norm2.Beta, up.W, up.B, down.W, down.B,
	} {
		if value.Dtype() != tensor.F32 || !value.IsContiguous() || value.Offset() != 0 {
			return false
		}
	}
	return true
}
