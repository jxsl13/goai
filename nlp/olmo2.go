package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// OLMo2 is Allen AI's OLMo 2 decoder-only transformer (transformers'
// Olmo2ForCausalLM). It shares Llama's building blocks — RMSNorm, rotary position
// embeddings (RoPE), grouped-query attention, a SwiGLU feed-forward, and NO biases —
// but reshuffles the residual structure and normalizes the query/key projections, which
// is why it is a SEPARATE type (like [Gemma2]) rather than a [Llama] variant:
//
//   - post-norm blocks: unlike Llama's pre-norm, OLMo 2 applies the sublayer to the RAW
//     residual stream and normalizes the sublayer OUTPUT before adding it back. There is
//     NO input_layernorm. Per block:
//
//     residual = x; a = attention(x);       a = post_attention_layernorm(a);  x = residual + a
//     residual = x; f = FFN(x);             f = post_feedforward_layernorm(f); x = residual + f
//
//   - full-width QK-norm: an RMSNorm is applied to the ENTIRE q_proj / k_proj output
//     (over num_heads·head_dim and num_kv_heads·head_dim respectively) BEFORE the split
//     into heads and RoPE — q = q_norm(Wq·x), k = k_norm(Wk·x). Because the norm spans
//     the full projection width (not per-head like Qwen3), it is a plain
//     [nn.RMSNorm.Forward] on the [seq, heads·head_dim] tensor.
//
// The RMSNorms are standard (γ·x̂, NOT Gemma's (1+γ) variant), the pre-softmax score
// scale is the usual 1/√head_dim, RoPE/GQA/SwiGLU are standard, and the LM head is
// untied by default (tie_word_embeddings=false). A final model.norm precedes the head.
// Load a Hugging Face Olmo2ForCausalLM checkpoint with [OLMo2FromHF].
type OLMo2 struct {
	Config OLMo2Config    // the model hyperparameters
	TokEmb *tensor.Tensor // [vocab, dim] token embedding (no positional embedding)
	Blocks []*OLMo2Block  // the stacked post-norm transformer blocks
	Norm   *nn.RMSNorm    // final pre-logits RMSNorm (model.norm)
	Out    *tensor.Tensor // [dim, vocab] output projection (tied to TokEmbᵀ when tie_word_embeddings)
}

// OLMo2Config fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred from
// the checkpoint by [OLMo2FromHF]; the remaining fields come from config.json. HeadDim
// is decoupled (config.head_dim) and inferred from q_proj when left 0.
type OLMo2Config struct {
	Vocab    int     // vocabulary size
	Ctx      int     // max context length
	Dim      int     // embedding width (d_model)
	Heads    int     // number of query heads
	KVHeads  int     // key/value heads (GQA); 0 → Heads (standard MHA)
	HeadDim  int     // per-head width (may differ from Dim/Heads); 0 → Dim/Heads
	Layers   int     // number of transformer blocks
	Hidden   int     // SwiGLU inner width
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE frequency base θ; 0 → 10000
}

// OLMo2Block is one post-norm OLMo 2 transformer block. The attention and FFN sublayers
// read the RAW residual stream; PostAttnNorm and PostFFNNorm normalize their OUTPUTS
// before the residual add. QNorm/KNorm are full-width RMSNorms over the q/k projections.
type OLMo2Block struct {
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (no bias); Wk/Wv are [dim, KVHeads·headDim]
	QNorm, KNorm   *nn.RMSNorm    // full-width q/k RMSNorm (over heads·headDim / kv·headDim) applied before RoPE
	PostAttnNorm   *nn.RMSNorm    // post_attention_layernorm (on the attention output, pre-residual)
	FFN            *nn.SwiGLU     // SwiGLU feed-forward
	PostFFNNorm    *nn.RMSNorm    // post_feedforward_layernorm (on the FFN output, pre-residual)
}

func (c OLMo2Config) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *OLMo2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// embed gathers the token embeddings for the prompt (OLMo 2 has no positional embedding —
// position enters through RoPE inside attention, and there is no embedding scale).
func (m *OLMo2) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: OLMo2 prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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

// hidden runs the post-norm block stack and the final RMSNorm on an embedding x
// [seq, dim], returning the pre-logits hidden states [seq, dim].
func (m *OLMo2) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenCapture(ctx, x, nil)
}

// hiddenCapture is hidden with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-k_norm, POST-RoPE k and raw v [seq, kvWidth] — exactly the rows
// [OLMo2.DecodeStep] appends per token, which is what lets [OLMo2.Prefill] seed
// a cache from ONE batched pass over the prompt. The tensors handed to capture
// are the same ones the ongoing forward consumes (callers must not mutate
// them; the cache copies rows into its own buffers). A nil capture is the
// plain forward: no extra ops are recorded, so tape/training semantics of
// hidden are byte-identical to before this hook existed.
func (m *OLMo2) hiddenCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for l, b := range m.Blocks {
		// Attention sublayer (post-norm): a = post_attention_layernorm(attn(x)); x = x + a.
		// Attention reads the RAW residual x — there is no input_layernorm.
		var tap func(k, v *tensor.Tensor)
		if capture != nil {
			tap = func(k, v *tensor.Tensor) { capture(l, k, v) }
		}
		a, err := m.attention(ctx, b, x, attn, tap)
		if err != nil {
			return nil, err
		}
		if a, err = b.PostAttnNorm.Forward(ctx, a); err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 residual OpAdd single backend op, matmul-dominated
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// FFN sublayer (post-norm): f = post_feedforward_layernorm(SwiGLU(x)); x = x + f.
		f, err := b.FFN.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if f, err = b.PostFFNNorm.Forward(ctx, f); err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 residual OpAdd single backend op, matmul-dominated
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	return m.Norm.Forward(ctx, x)
}

// attention computes OLMo 2's multi-head attention over the raw residual x [seq, dim]:
// project q/k/v, apply the full-width q_norm/k_norm to the WHOLE q/k projections (before
// the head split), RoPE, then standard causal GQA via OpMHA and the o_proj. Mirrors
// Olmo2Attention.forward exactly. capture, when non-nil, receives the post-k_norm
// POST-RoPE k and the raw v — the per-token cache rows — for [OLMo2.Prefill].
func (m *OLMo2) attention(ctx *backend.Context, b *OLMo2Block, x *tensor.Tensor, attn backend.AttnAttrs, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	q, err := project(ctx, x, b.Wq)
	if err != nil {
		return nil, err
	}
	k, err := project(ctx, x, b.Wk)
	if err != nil {
		return nil, err
	}
	v, err := project(ctx, x, b.Wv)
	if err != nil {
		return nil, err
	}
	// Full-width QK-norm: RMSNorm over the ENTIRE q/k projection (last axis = heads·headDim
	// for q, kv·headDim for k), applied BEFORE the head split and RoPE.
	if q, err = b.QNorm.Forward(ctx, q); err != nil {
		return nil, err
	}
	if k, err = b.KNorm.Forward(ctx, k); err != nil {
		return nil, err
	}
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, k); err != nil {
		return nil, err
	}
	if capture != nil {
		capture(k, v)
	}
	a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return project(ctx, a, b.Wo)
}

// Params returns every trainable tensor for optimizers.
func (m *OLMo2) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wo)
		ps = append(ps, b.QNorm.Gamma, b.KNorm.Gamma, b.PostAttnNorm.Gamma, b.PostFFNNorm.Gamma)
		ps = append(ps, b.FFN.Params()...)
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
