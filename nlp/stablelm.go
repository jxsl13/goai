package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// StableLM is Stability AI's StableLM decoder-only transformer (transformers'
// StableLmForCausalLM). It is a Llama-shaped decoder — RoPE, grouped-query attention, a
// SwiGLU feed-forward, untied LM head — but takes THREE departures that make it a SEPARATE
// type (like [Cohere] / [OLMo2]) rather than a [Llama] variant:
//
//   - LayerNorm WITH bias (not RMSNorm): input_layernorm, post_attention_layernorm and the
//     final model.norm are ordinary nn.LayerNorm(hidden_size) with BOTH a weight (γ) and a
//     bias (β). GoAI loads both from the checkpoint (see [StableLMFromHF]); β is NOT forced
//     to zero (that is the [Cohere] weight-only case). Getting this wrong — treating the
//     norm as weight-only or as an RMSNorm — moves the logits far off the golden.
//
//   - SEQUENTIAL residual (use_parallel_residual=False, the default): the classic Llama
//     two-norm block, NOT Cohere's parallel one-norm block:
//
//     x = x + attention(input_layernorm(x)); x = x + FFN(post_attention_layernorm(x))
//
//   - PARTIAL rotary (partial_rotary_factor, e.g. 0.25): only the first rotaryDim channels
//     of each head are rotated, the rest pass through unrotated. rotaryDim =
//     int(head_dim · partial_rotary_factor). StableLM's rotate_half is the split-half
//     convention — the same as GoAI's OpRoPE — so [partialRoPE] applies directly with no
//     interleave permutation (unlike Cohere). See [partialRoPE].
//
// StableLM's default config has NO attention bias (use_qkv_bias=false) and NO QK-norm
// (qk_layernorm=false); GoAI builds for those defaults. Attention uses the standard
// 1/√head_dim score scale, supports GQA (num_key_value_heads) and an UNTIED LM head
// (lm_head.weight, present in the checkpoint). Load a Hugging Face StableLmForCausalLM
// checkpoint with [StableLMFromHF].
type StableLM struct {
	Config StableLMConfig   // the model hyperparameters
	TokEmb *tensor.Tensor   // [vocab, dim] token embedding (no positional embedding)
	Blocks []*StableLMBlock // the stacked sequential-residual transformer blocks
	Norm   *nn.LayerNorm    // final pre-logits LayerNorm (model.norm), weight AND bias
	Out    *tensor.Tensor   // [dim, vocab] output projection (untied lm_head.weightᵀ)
}

// StableLMConfig fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred from
// the checkpoint by [StableLMFromHF]; the remaining fields come from config.json. The
// rotary width is set either directly (RotaryDim) or as a fraction (RotaryPct,
// config.partial_rotary_factor); RotaryDim wins when both are set.
type StableLMConfig struct {
	Vocab     int     // vocabulary size
	Ctx       int     // max context length
	Dim       int     // embedding width (d_model)
	Heads     int     // number of query heads
	KVHeads   int     // key/value heads (GQA); 0 → Heads (standard MHA)
	HeadDim   int     // per-head width; 0 → Dim/Heads
	Layers    int     // number of transformer blocks
	Hidden    int     // SwiGLU inner width
	Eps       float64 // LayerNorm epsilon (layer_norm_eps, ~1e-5)
	RopeBase  float64 // RoPE frequency base θ; 0 → 10000
	RotaryDim int     // rotated channels per head; 0 → int(headDim·RotaryPct)
	RotaryPct float64 // partial_rotary_factor; used when RotaryDim==0; 0 → 0.25
}

// StableLMBlock is one sequential-residual StableLM transformer block. Unlike Cohere's
// single parallel norm, it carries TWO LayerNorms (each with weight and bias): InputNorm
// gates the attention sublayer and PostAttnNorm gates the FFN, each on its own residual add.
type StableLMBlock struct {
	InputNorm      *nn.LayerNorm  // input_layernorm (weight+bias), feeds attention
	PostAttnNorm   *nn.LayerNorm  // post_attention_layernorm (weight+bias), feeds FFN
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (no bias in the default config)
	FFN            *nn.SwiGLU     // SwiGLU feed-forward (silu gate)
}

func (c StableLMConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// headDim is the per-head width (config.head_dim); StableLM does not decouple it, so it
// defaults to Dim/Heads.
func (c StableLMConfig) headDim() int {
	if c.HeadDim > 0 {
		return c.HeadDim
	}
	return c.Dim / c.Heads
}

// rotaryDim is the number of channels per head that carry RoPE (partial rotary). It is
// RotaryDim when set, else int(head_dim · RotaryPct) with RotaryPct defaulting to 0.25.
func (c StableLMConfig) rotaryDim() int {
	if c.RotaryDim > 0 {
		return c.RotaryDim
	}
	pct := c.RotaryPct
	if pct == 0 {
		pct = 0.25
	}
	return int(float64(c.headDim()) * pct)
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *StableLM) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// embed gathers the token embeddings for the prompt (StableLM has no positional embedding —
// position enters through RoPE inside attention, and there is no embedding scale).
func (m *StableLM) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: StableLM prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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
func (m *StableLM) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenCapture(ctx, x, nil)
}

// hiddenCapture is hidden with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-partial-RoPE k and raw post-projection v [seq, kvWidth] — exactly the
// rows a KV-cache decode appends per token ([StableLM.DecodeStep]), which is
// what lets [StableLM.Prefill] seed a cache from ONE batched pass over the
// prompt. The tensors handed to capture are the same ones the ongoing forward
// consumes (callers must not mutate them; the cache copies rows into its own
// buffers). A nil capture is the plain forward: no extra ops are recorded, so
// the semantics of hidden are byte-identical to before this hook existed.
func (m *StableLM) hiddenCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for l, b := range m.Blocks {
		// Sequential residual (Llama-style): input_layernorm → attention → add; then
		// post_attention_layernorm → FFN → add.
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var tap func(k, v *tensor.Tensor)
		if capture != nil {
			tap = func(k, v *tensor.Tensor) { capture(l, k, v) }
		}
		a, err := m.attention(ctx, b, xn, attn, tap)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 residual OpAdd single backend op, matmul-dominated
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		xn2, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := b.FFN.Forward(ctx, xn2)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 residual OpAdd single backend op, matmul-dominated
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	return m.Norm.Forward(ctx, x)
}

// attention computes StableLM's multi-head attention over the normalized input xn [seq,
// dim]: project q/k/v, apply PARTIAL split-half RoPE (only the first rotaryDim channels of
// each head rotate), then standard causal GQA via OpMHA and the o_proj. Mirrors
// StableLmAttention.forward for the default config (no qkv bias, no QK-norm). A non-nil
// capture receives the post-partial-RoPE k and raw v right before OpMHA (the Prefill KV
// tap, see [StableLM.hiddenCapture]); nil is the plain forward.
func (m *StableLM) attention(ctx *backend.Context, b *StableLMBlock, xn *tensor.Tensor, attn backend.AttnAttrs, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	rot := cfg.rotaryDim()
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
	if q, err = partialRoPE(ctx, q, cfg.Heads, rot, backend.RoPEAttrs{Base: cfg.RopeBase}); err != nil {
		return nil, err
	}
	if k, err = partialRoPE(ctx, k, kv, rot, backend.RoPEAttrs{Base: cfg.RopeBase}); err != nil {
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
func (m *StableLM) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wo)
		ps = append(ps, b.InputNorm.Params()...)
		ps = append(ps, b.PostAttnNorm.Params()...)
		ps = append(ps, b.FFN.Params()...)
	}
	ps = append(ps, m.Norm.Params()...)
	ps = append(ps, m.Out)
	return ps
}
