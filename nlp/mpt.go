package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MPT is MosaicML's MPT decoder-only transformer (transformers' MptForCausalLM). It is a
// separate type — not a [Llama] or [GPTNeoX] variant — because it is the library's first
// ALiBi model and takes four departures from the Llama/NeoX blocks, all confirmed against
// transformers.models.mpt.modeling_mpt:
//
//   - ALiBi position bias, NO RoPE and NO positional embedding. Instead of rotating the
//     query/key channels, MPT adds a static per-head linear-distance penalty to the
//     pre-softmax attention scores (Press et al. 2021, arXiv:2108.12409). Head h gets slope
//     2^(−8·(h+1)/n) for n heads (MptConfig's alibi_bias_max=8 default). GoAI's [backend.OpMHA]
//     applies exactly this bias when AttnAttrs.ALiBi is set — the ref kernel adds
//     slopeₕ·(j−iₐᵦₛ) (relative distance) while transformers adds slopeₕ·(j−(seq−1)) (key
//     index only), but the two differ by a per-query-row constant that softmax is invariant
//     to, so they yield identical attention weights. The slopes match [backend.ALiBiSlopes]
//     bit-for-bit for a power-of-two head count.
//
//   - WEIGHT-ONLY LayerNorm (no_bias=True). norm_1, norm_2 and norm_f are LPLayerNorm
//     modules carrying only γ; the mean-centered LayerNorm runs with β = 0 (loaded via
//     [layerNormFromHF]).
//
//   - PACKED, standard-MHA Wqkv with NO bias. attn.Wqkv is [3·hidden, hidden] and is split by
//     transformers' mixed_qkv.chunk(3) into contiguous [all-q; all-k; all-v] blocks — the
//     Phi-3 layout, NOT the GPT-NeoX per-head interleave — so [sliceRows] recovers q/k/v.
//     out_proj is likewise bias-free.
//
//   - GELU MLP, NO bias, NO gate. ffn.up_proj → exact-erf GELU → ffn.down_proj, both Linear
//     layers bias-free (matching GoAI's OpGELU; transformers uses nn.GELU(approximate="none")).
//
// The residual is SEQUENTIAL (x = x + attn(norm_1(x)); x = x + ffn(norm_2(x))), attention
// uses the standard 1/√head_dim score scale and full MHA (no GQA), and the LM head is TIED to
// the token embedding (logits = hidden · wteᵀ). Load a Hugging Face MptForCausalLM checkpoint
// with [MPTFromHF].
type MPT struct {
	Config    MPTConfig      // the model hyperparameters
	TokEmb    *tensor.Tensor // [vocab, dim] token embedding (transformer.wte)
	Blocks    []*MPTBlock    // the stacked sequential-residual transformer blocks
	FinalNorm *nn.LayerNorm  // transformer.norm_f (weight-only LayerNorm)
	Out       *tensor.Tensor // [dim, vocab] output projection (tied wteᵀ)
}

// MPTConfig fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred from the
// checkpoint by [MPTFromHF]; Heads, Eps and Ctx come from config.json.
type MPTConfig struct {
	Vocab  int     // vocabulary size
	Ctx    int     // max context length (max_seq_len)
	Dim    int     // embedding width (d_model)
	Heads  int     // number of attention heads (MPT is MHA: no GQA)
	Layers int     // number of transformer blocks
	Hidden int     // MLP inner width (expansion_ratio · d_model, typically 4·d_model)
	Eps    float64 // LayerNorm epsilon (layer_norm_epsilon, ~1e-5)
}

// MPTBlock is one sequential-residual MPT transformer block. Every projection is bias-free
// and both norms are weight-only LayerNorms.
type MPTBlock struct {
	Norm1, Norm2   *nn.LayerNorm  // norm_1 (pre-attention) / norm_2 (pre-MLP), weight-only
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (Wqkv split + out_proj), no bias
	Wup, Wdown     *tensor.Tensor // ffn.up_proj / ffn.down_proj, no bias
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *MPT) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// embed gathers the token embeddings for the prompt. MPT has no positional embedding and no
// embedding scale — position enters solely through the ALiBi bias inside attention.
func (m *MPT) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: MPT prompt length %d outside (0,%d]", seq, m.Config.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	return exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
}

// hidden runs the sequential-residual block stack and the final LayerNorm on an embedding x
// [seq, dim], returning the pre-logits hidden states [seq, dim].
func (m *MPT) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	cfg := m.Config
	attn := backend.AttnAttrs{Heads: cfg.Heads, Causal: true, ALiBi: true}
	for _, b := range m.Blocks {
		// Sequential residual: attention over norm_1(x) folds back onto x, then the MLP over
		// norm_2(x) folds onto the updated x.
		an, err := b.Norm1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.attention(ctx, b, an, attn)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		fn, err := b.Norm2.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, fn)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	return m.FinalNorm.Forward(ctx, x)
}

// attention computes MPT multi-head attention over the normalized input xn [seq, dim]:
// project the bias-free q/k/v, run causal MHA with the per-head ALiBi bias (no RoPE), then the
// bias-free out_proj. Mirrors MptAttention.forward.
func (m *MPT) attention(ctx *backend.Context, b *MPTBlock, xn *tensor.Tensor, attn backend.AttnAttrs) (*tensor.Tensor, error) {
	q, err := project(ctx, xn, b.Wq)
	if err != nil {
		return nil, err
	}
	k, err := project(ctx, xn, b.Wk)
	if err != nil {
		return nil, err
	}
	v, err := project(ctx, xn, b.Wv)
	if err != nil {
		return nil, err
	}
	a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return project(ctx, a, b.Wo)
}

// mlp runs MPT's bias-free GELU feed-forward on the normalized input fn [seq, dim]: up_proj →
// exact-erf GELU → down_proj. Mirrors MptMLP.forward.
func (m *MPT) mlp(ctx *backend.Context, b *MPTBlock, fn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := project(ctx, fn, b.Wup)
	if err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return project(ctx, h, b.Wdown)
}

// Params returns every trainable tensor for optimizers. The tied LM head shares TokEmb's
// storage, so Out is deliberately omitted to avoid a double update.
func (m *MPT) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wo, b.Wup, b.Wdown)
		ps = append(ps, b.Norm1.Params()...)
		ps = append(ps, b.Norm2.Params()...)
	}
	ps = append(ps, m.FinalNorm.Params()...)
	return ps
}
