package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// StarCoder2 is BigCode's StarCoder2 decoder-only transformer (transformers'
// Starcoder2ForCausalLM). It is a Llama-shaped decoder — RoPE, grouped-query attention, an
// untied LM head — but takes FOUR departures from the Llama block that make it a SEPARATE
// type (like [StableLM] / [GPTNeoX]) rather than a [Llama] variant. All four are confirmed
// against transformers.models.starcoder2.modeling_starcoder2:
//
//   - LayerNorm WITH bias (not RMSNorm). input_layernorm, post_attention_layernorm and the
//     final model.norm are ordinary nn.LayerNorm(hidden_size, eps=config.norm_epsilon)
//     carrying BOTH a weight (γ) and a bias (β), both loaded from the checkpoint (see
//     [StarCoder2FromHF]). Treating the norm as weight-only or as an RMSNorm moves the logits
//     far off the golden — this is the same full-LayerNorm as [StableLM] / [GPTNeoX].
//
//   - SEQUENTIAL residual (like Llama / [StableLM], NOT [GPTNeoX]'s parallel one): the classic
//     two-norm block —
//
//     x = x + attention(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x))
//
//   - BIASED attention (config.use_bias=True). q_proj / k_proj / v_proj / o_proj ALL carry a
//     bias, added right after each projection (see [StarCoder2.attention]). GQA is supported
//     (num_key_value_heads). Rotary is FULL (not partial) — the standard split-half OpRoPE
//     over all head_dim channels, like [Llama] — with the standard 1/√head_dim score scale.
//
//   - GELU MLP with bias, 2-layer (NOT SwiGLU): mlp.c_fc (biased) → activation → mlp.c_proj
//     (biased). hidden_act is "gelu_pytorch_tanh" (the tanh approximation), whereas GoAI's
//     OpGELU is the exact erf GELU; this leaves a small approximation residual (like [Gemma]),
//     measured well within the 2e-3 forward-parity gate. See [StarCoder2.mlp].
//
// The LM head is UNTIED (tie_word_embeddings=false): lm_head.weight is present in the
// checkpoint and used directly. Load a Hugging Face Starcoder2ForCausalLM checkpoint with
// [StarCoder2FromHF].
type StarCoder2 struct {
	Config StarCoder2Config   // the model hyperparameters
	TokEmb *tensor.Tensor     // [vocab, dim] token embedding (model.embed_tokens, no positional embedding)
	Blocks []*StarCoder2Block // the stacked sequential-residual transformer blocks
	Norm   *nn.LayerNorm      // final pre-logits LayerNorm (model.norm), weight AND bias
	Out    *tensor.Tensor     // [dim, vocab] output projection (untied lm_head.weightᵀ)
}

// StarCoder2Config fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred from
// the checkpoint by [StarCoder2FromHF]; the remaining fields come from config.json.
type StarCoder2Config struct {
	Vocab    int     // vocabulary size
	Ctx      int     // max context length (max_position_embeddings)
	Dim      int     // embedding width (hidden_size)
	Heads    int     // number of query heads (num_attention_heads)
	KVHeads  int     // key/value heads (GQA, num_key_value_heads); 0 → Heads (standard MHA)
	HeadDim  int     // per-head width; 0 → Dim/Heads (StarCoder2 does not decouple it)
	Layers   int     // number of transformer blocks
	Hidden   int     // MLP inner width (intermediate_size)
	Eps      float64 // LayerNorm epsilon (norm_epsilon, ~1e-5)
	RopeBase float64 // RoPE frequency base θ (rope_theta); 0 → 10000
}

// StarCoder2Block is one sequential-residual StarCoder2 transformer block. It carries TWO full
// LayerNorms (each with weight and bias): InputNorm gates the attention sublayer and
// PostAttnNorm gates the MLP, each on its own residual add. Every attention projection carries
// a bias, and the MLP is a biased 2-layer GELU (c_fc → GELU → c_proj), not a SwiGLU.
type StarCoder2Block struct {
	InputNorm      *nn.LayerNorm  // input_layernorm (weight+bias), feeds attention
	PostAttnNorm   *nn.LayerNorm  // post_attention_layernorm (weight+bias), feeds the MLP
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (q/k/v/o_proj), all biased
	Bq, Bk, Bv, Bo *tensor.Tensor // their biases (use_bias=True)
	Wfc, Bfc       *tensor.Tensor // mlp.c_fc weight and bias (dim → hidden)
	Wproj, Bproj   *tensor.Tensor // mlp.c_proj weight and bias (hidden → dim)
}

func (c StarCoder2Config) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *StarCoder2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// embed gathers the token embeddings for the prompt (StarCoder2 has no positional embedding —
// position enters through the full RoPE inside attention, and there is no embedding scale).
func (m *StarCoder2) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: StarCoder2 prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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
func (m *StarCoder2) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenCapture(ctx, x, nil)
}

// hiddenCapture is hidden with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-bias, post-RoPE k and post-bias v [seq, kvWidth] — exactly the rows a
// KV-cache decode appends per token ([StarCoder2.DecodeStep]), which is what
// lets [StarCoder2.Prefill] seed a cache from ONE batched pass over the prompt.
// The tensors handed to capture are the same ones the ongoing forward consumes
// (callers must not mutate them; the cache copies rows into its own buffers).
// A nil capture is the plain forward: no extra ops are recorded, so the
// semantics of hidden are byte-identical to before this hook existed.
func (m *StarCoder2) hiddenCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: cfg.kvHeads(), Causal: true}
	for l, b := range m.Blocks {
		// Sequential residual (Llama-style): input_layernorm → attention → add; then
		// post_attention_layernorm → MLP → add.
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var tap func(k, v *tensor.Tensor)
		if capture != nil {
			tap = func(k, v *tensor.Tensor) { capture(l, k, v) }
		}
		a, err := m.attention(ctx, b, xn, attn, 0, tap)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		xn2, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, xn2)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	return m.Norm.Forward(ctx, x)
}

// attention computes StarCoder2's multi-head attention over the normalized input xn [seq,
// dim]: project the biased q/k/v, apply FULL split-half RoPE at position offset pos, then
// standard causal GQA via OpMHA and the biased o_proj. Mirrors Starcoder2Attention.forward.
// A non-nil capture receives the post-RoPE k and post-bias v right before OpMHA (the
// Prefill KV tap, see [StarCoder2.hiddenCapture]); nil is the plain forward.
func (m *StarCoder2) attention(ctx *backend.Context, b *StarCoder2Block, xn *tensor.Tensor, attn backend.AttnAttrs, pos int, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	q, err := projBias(ctx, xn, b.Wq, b.Bq)
	if err != nil {
		return nil, err
	}
	k, err := projBias(ctx, xn, b.Wk, b.Bk)
	if err != nil {
		return nil, err
	}
	v, err := projBias(ctx, xn, b.Wv, b.Bv)
	if err != nil {
		return nil, err
	}
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}, k); err != nil {
		return nil, err
	}
	if capture != nil {
		capture(k, v)
	}
	a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return projBias(ctx, a, b.Wo, b.Bo)
}

// mlp runs StarCoder2's biased 2-layer GELU feed-forward on the normalized input fn [seq,
// dim]: c_fc → GELU → c_proj. Mirrors Starcoder2MLP.forward. The reference activation is
// "gelu_pytorch_tanh" (tanh approximation); GoAI's OpGELU is exact-erf, leaving a small
// residual that stays inside the 2e-3 forward-parity gate.
func (m *StarCoder2) mlp(ctx *backend.Context, b *StarCoder2Block, fn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := projBias(ctx, fn, b.Wfc, b.Bfc)
	if err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return projBias(ctx, h, b.Wproj, b.Bproj)
}

// Params returns every trainable tensor for optimizers.
func (m *StarCoder2) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wo, b.Bq, b.Bk, b.Bv, b.Bo, b.Wfc, b.Bfc, b.Wproj, b.Bproj)
		ps = append(ps, b.InputNorm.Params()...)
		ps = append(ps, b.PostAttnNorm.Params()...)
	}
	ps = append(ps, m.Norm.Params()...)
	ps = append(ps, m.Out)
	return ps
}
