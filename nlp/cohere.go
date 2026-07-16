package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Cohere is Cohere's Command-R decoder-only transformer (transformers'
// CohereForCausalLM). It shares Llama's building blocks — RoPE, grouped-query
// attention, a SwiGLU feed-forward, NO biases and tied embeddings — but takes FOUR
// departures that make it a SEPARATE type (like [OLMo2] / [Gemma2]) rather than a
// [Llama] variant:
//
//   - PARALLEL residual (not sequential): each block reads the residual stream through
//     ONE norm (input_layernorm) that feeds BOTH the attention and the FFN, and SUMS
//     their outputs onto the residual. There is no separate FFN norm and no post-norm:
//
//     residual = x; xn = input_layernorm(x); x = residual + attention(xn) + FFN(xn)
//
//   - mean-centered, weight-only LayerNorm (not RMSNorm): CohereLayerNorm subtracts the
//     mean, divides by √(var+eps) and multiplies by a weight — no bias. GoAI models it
//     with [nn.LayerNorm] (a proper mean-centered LayerNorm with γ and β) whose β is a
//     ZERO vector, so it reduces to weight-only. This applies to every norm: each block's
//     InputNorm and the final model.norm.
//
//   - INTERLEAVED RoPE (GPT-J style): Cohere's rotate_half pairs even/odd channels
//     (x[..., ::2] with x[..., 1::2]) instead of Llama's split-half pairing. GoAI's
//     OpRoPE is split-half, so [CohereFromHF] PERMUTES the q_proj and k_proj weight rows
//     per head at load time (interleaved → split-half layout); split-half OpRoPE on the
//     permuted weights then reproduces Cohere's interleaved rotary exactly. V and O are
//     left untouched. See [CohereFromHF] for the permutation.
//
//   - logit_scale: the final logits are multiplied by a constant (config.logit_scale,
//     e.g. 0.0625) before the softmax. The LM head is tied to the embedding table.
//
// Attention uses the standard 1/√head_dim score scale and Cohere v1 has NO QK-norm
// (use_qk_norm=false). Load a Hugging Face CohereForCausalLM checkpoint with
// [CohereFromHF].
type Cohere struct {
	Config CohereConfig   // the model hyperparameters
	TokEmb *tensor.Tensor // [vocab, dim] token embedding (no positional embedding)
	Blocks []*CohereBlock // the stacked parallel-residual transformer blocks
	Norm   *nn.LayerNorm  // final pre-logits LayerNorm (model.norm), weight-only (β=0)
	Out    *tensor.Tensor // [dim, vocab] output projection (tied to TokEmbᵀ)
}

// CohereConfig fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred from
// the checkpoint by [CohereFromHF]; the remaining fields come from config.json. HeadDim
// is decoupled (config.head_dim) and inferred from q_proj when left 0.
type CohereConfig struct {
	Vocab      int     // vocabulary size
	Ctx        int     // max context length
	Dim        int     // embedding width (d_model)
	Heads      int     // number of query heads
	KVHeads    int     // key/value heads (GQA); 0 → Heads (standard MHA)
	HeadDim    int     // per-head width (may differ from Dim/Heads); 0 → Dim/Heads
	Layers     int     // number of transformer blocks
	Hidden     int     // SwiGLU inner width
	Eps        float64 // LayerNorm epsilon (layer_norm_eps, ~1e-5)
	RopeBase   float64 // RoPE frequency base θ; 0 → 10000
	LogitScale float64 // final-logit multiplier (config.logit_scale, e.g. 0.0625); 0 → 1
}

// CohereBlock is one parallel-residual Command-R transformer block. InputNorm is the
// single per-block LayerNorm read by BOTH sublayers; the attention and FFN outputs are
// summed onto the raw residual. There is no post-norm and no separate FFN norm.
type CohereBlock struct {
	InputNorm      *nn.LayerNorm  // input_layernorm (weight-only, β=0), feeds attention AND FFN
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (no bias); Wq/Wk carry the interleaved→split-half RoPE row permutation
	FFN            *nn.SwiGLU     // SwiGLU feed-forward
}

func (c CohereConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

func (c CohereConfig) logitScale() float64 {
	if c.LogitScale == 0 {
		return 1
	}
	return c.LogitScale
}

// Forward computes logits [seq, vocab] for the prompt tokens, scaled by logit_scale.
func (m *Cohere) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.Out)
	if err != nil {
		return nil, err
	}
	return m.scaleLogits(ctx, logits)
}

// scaleLogits multiplies the raw logits by config.logit_scale (Command-R's final
// tempering; e.g. 0.0625). A scale of 1 is returned unchanged.
func (m *Cohere) scaleLogits(ctx *backend.Context, logits *tensor.Tensor) (*tensor.Tensor, error) {
	s := m.Config.logitScale()
	if s == 1 {
		return logits, nil
	}
	scale := tensor.New(tensor.F64, tensor.Shape{})
	scale.Storage().F64()[0] = s
	return exec1(ctx, backend.OpMul, nil, logits, scale)
}

// embed gathers the token embeddings for the prompt (Command-R has no positional
// embedding — position enters through RoPE inside attention, and there is no embedding
// scale).
func (m *Cohere) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Cohere prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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

// hidden runs the parallel-residual block stack and the final LayerNorm on an embedding x
// [seq, dim], returning the pre-logits hidden states [seq, dim].
func (m *Cohere) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for _, b := range m.Blocks {
		// Parallel residual: ONE norm feeds both sublayers; their outputs are summed onto
		// the raw residual. xn = input_layernorm(x); x = x + attention(xn) + FFN(xn).
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.attention(ctx, b, xn, attn)
		if err != nil {
			return nil, err
		}
		f, err := b.FFN.Forward(ctx, xn)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	return m.Norm.Forward(ctx, x)
}

// attention computes Command-R's multi-head attention over the normalized input xn
// [seq, dim]: project q/k/v, apply the split-half OpRoPE (which reproduces Cohere's
// interleaved rotary because Wq/Wk were row-permuted at load), then standard causal GQA
// via OpMHA and the o_proj. Mirrors CohereAttention.forward (no QK-norm in Cohere v1).
func (m *Cohere) attention(ctx *backend.Context, b *CohereBlock, xn *tensor.Tensor, attn backend.AttnAttrs) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
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
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, k); err != nil {
		return nil, err
	}
	a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return project(ctx, a, b.Wo)
}

// Params returns every trainable tensor for optimizers.
func (m *Cohere) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wo)
		ps = append(ps, b.InputNorm.Params()...)
		ps = append(ps, b.FFN.Params()...)
	}
	ps = append(ps, m.Norm.Params()...)
	ps = append(ps, m.Out)
	return ps
}
