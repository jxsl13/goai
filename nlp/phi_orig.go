package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Phi is Microsoft's ORIGINAL Phi decoder-only transformer — Phi-1, Phi-1.5 and Phi-2
// (transformers' PhiForCausalLM). It is NOT Phi-3 (that Llama-shaped, RMSNorm/SwiGLU/packed-
// QKV model is loaded by [Phi3FromHF]); the two share only a name. Original Phi is a
// GPT-NeoX-flavoured block that fuses the [Cohere] single-norm parallel residual with the
// [GPTNeoX] biased-everywhere / LayerNorm / partial-rotary machinery, so it is a SEPARATE
// type. Every departure below is confirmed against transformers.models.phi.modeling_phi:
//
//   - SINGLE-norm PARALLEL residual (like [Cohere], NOT GPT-NeoX's two norms):
//     PhiDecoderLayer.forward reads the residual stream through ONE input_layernorm that
//     feeds BOTH the attention and the MLP, and sums their outputs onto the raw residual:
//
//     residual = x; xn = input_layernorm(x); x = attention(xn) + mlp(xn) + residual
//
//   - Full LayerNorm WITH bias (not RMSNorm, not weight-only): each block's input_layernorm
//     and the model's final_layernorm are ordinary nn.LayerNorm carrying both γ and β,
//     loaded from the checkpoint (see [GPTNeoX] for the same LayerNorm-with-bias pattern).
//
//   - SEPARATE q/k/v projections WITH bias (Phi does NOT pack qkv like GPT-NeoX):
//     self_attn.{q,k,v}_proj and the self_attn.dense output projection are all
//     nn.Linear(..., bias=True). No de-interleave split is needed — the state_dict already
//     stores head-contiguous q/k/v. The default config has qk_layernorm=false (no QK-norm),
//     which GoAI builds for.
//
//   - PARTIAL rotary (partial_rotary_factor=0.5): only rotaryDim = int(head_dim ·
//     partial_rotary_factor) channels of each head are rotated, the rest pass through. Phi's
//     rotate_half is split-half — the same convention as GoAI's OpRoPE — so [partialRoPE]
//     applies directly with no interleave permutation (contrast [Cohere]).
//
//   - GELU MLP with bias: mlp.fc1 → gelu_new → mlp.fc2, both Linear layers carrying a bias.
//     Phi's default hidden_act is "gelu_new" (the TANH approximation); GoAI's OpGELU is the
//     EXACT erf GELU, so the loaded model reproduces transformers' logits only up to a small
//     approximation residual (the same exact-vs-tanh gap as T5 v1.1 / Gemma). The parity
//     anchor in phi_orig_hf_test.go measures and gates that residual.
//
// Attention uses the standard 1/√head_dim score scale and MHA (num_key_value_heads ==
// num_attention_heads by default). The LM head is UNTIED and — unlike most models — carries a
// BIAS (lm_head.weight AND lm_head.bias), which is added to the final logits. Load a Hugging
// Face PhiForCausalLM (Phi-1/1.5/2) checkpoint with [PhiFromHF].
type Phi struct {
	Config    PhiConfig      // the model hyperparameters
	TokEmb    *tensor.Tensor // [vocab, dim] token embedding (model.embed_tokens)
	Blocks    []*PhiBlock    // the stacked single-norm parallel-residual transformer blocks
	FinalNorm *nn.LayerNorm  // model.final_layernorm (LayerNorm with bias)
	Out       *tensor.Tensor // [dim, vocab] output projection (lm_head.weightᵀ, untied)
	OutBias   *tensor.Tensor // [vocab] output bias (lm_head.bias) added to the logits
}

// PhiConfig fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred from the
// checkpoint by [PhiFromHF]; the remaining fields come from config.json. The rotary width is
// set either directly (RotaryDim) or as a fraction (RotaryPct, config.partial_rotary_factor);
// RotaryDim wins when both are set.
type PhiConfig struct {
	Vocab     int     // vocabulary size
	Ctx       int     // max context length (max_position_embeddings)
	Dim       int     // embedding width (hidden_size)
	Heads     int     // number of query heads (num_attention_heads)
	KVHeads   int     // key/value heads (GQA); 0 → Heads (standard MHA, Phi's default)
	HeadDim   int     // per-head width; 0 → Dim/Heads
	Layers    int     // number of transformer blocks
	Hidden    int     // MLP inner width (intermediate_size)
	Eps       float64 // LayerNorm epsilon (layer_norm_eps, ~1e-5)
	RopeBase  float64 // RoPE frequency base θ (rope_theta); 0 → 10000
	RotaryDim int     // rotated channels per head; 0 → int(headDim·RotaryPct)
	RotaryPct float64 // partial_rotary_factor; used when RotaryDim==0; 0 → 0.5
}

// PhiBlock is one single-norm parallel-residual Phi transformer block. InputNorm is the
// single per-block LayerNorm (weight+bias) read by BOTH sublayers; the attention and MLP
// outputs are summed onto the raw residual. Every projection carries a bias.
type PhiBlock struct {
	InputNorm          *nn.LayerNorm  // input_layernorm (weight+bias), feeds attention AND MLP
	Wq, Wk, Wv, Wdense *tensor.Tensor // attention projections (q/k/v_proj + dense)
	Bq, Bk, Bv, Bdense *tensor.Tensor // their biases
	Wfc1, Bfc1         *tensor.Tensor // mlp.fc1 weight and bias
	Wfc2, Bfc2         *tensor.Tensor // mlp.fc2 weight and bias
}

func (c PhiConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// headDim is the per-head width; Phi uses head_dim = Dim/Heads unless decoupled.
func (c PhiConfig) headDim() int {
	if c.HeadDim > 0 {
		return c.HeadDim
	}
	return c.Dim / c.Heads
}

// rotaryDim is the number of channels per head that carry RoPE (partial rotary). It is
// RotaryDim when set, else int(head_dim · RotaryPct) with RotaryPct defaulting to 0.5 (Phi's
// partial_rotary_factor default).
func (c PhiConfig) rotaryDim() int {
	if c.RotaryDim > 0 {
		return c.RotaryDim
	}
	pct := c.RotaryPct
	if pct == 0 {
		pct = 0.5
	}
	return int(float64(c.headDim()) * pct)
}

// Forward computes logits [seq, vocab] for the prompt tokens (lm_head, with its bias).
func (m *Phi) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return m.logits(ctx, x)
}

// logits applies the untied, biased LM head: matmul(x, lm_head.weightᵀ) + lm_head.bias.
func (m *Phi) logits(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	l, err := exec1(ctx, backend.OpMatMul, nil, x, m.Out)
	if err != nil {
		return nil, err
	}
	return addBiasIf(ctx, l, m.OutBias)
}

// embed gathers the token embeddings for the prompt (Phi has no positional embedding and no
// embedding scale — position enters through the partial RoPE inside attention).
func (m *Phi) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Phi prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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

// hidden runs the single-norm parallel-residual block stack and the final LayerNorm on an
// embedding x [seq, dim], returning the pre-logits hidden states [seq, dim].
func (m *Phi) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenCapture(ctx, x, nil)
}

// hiddenCapture is hidden with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-bias, post-partial-RoPE k and post-bias v [seq, kvWidth] — exactly the
// rows a KV-cache decode appends per token ([Phi.DecodeStep]), which is what
// lets [Phi.Prefill] seed a cache from ONE batched pass over the prompt. The
// tensors handed to capture are the same ones the ongoing forward consumes
// (callers must not mutate them; the cache copies rows into its own buffers).
// A nil capture is the plain forward: no extra ops are recorded, so the
// semantics of hidden are byte-identical to before this hook existed.
func (m *Phi) hiddenCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: cfg.kvHeads(), Causal: true}
	for l, b := range m.Blocks {
		// Parallel residual: ONE input_layernorm feeds both sublayers; their outputs are summed
		// onto the raw residual. xn = input_layernorm(x); x = attention(xn) + mlp(xn) + x.
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
		f, err := m.mlp(ctx, b, xn)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	return m.FinalNorm.Forward(ctx, x)
}

// attention computes Phi's multi-head attention over the normalized input xn [seq, dim]:
// project the biased q/k/v, apply the PARTIAL split-half RoPE at position offset pos, then
// standard causal MHA and the biased output dense. Mirrors PhiAttention.forward for the
// default config (no QK-norm). A non-nil capture receives the post-partial-RoPE k and
// post-bias v right before OpMHA (the Prefill KV tap, see [Phi.hiddenCapture]); nil is the
// plain forward.
func (m *Phi) attention(ctx *backend.Context, b *PhiBlock, xn *tensor.Tensor, attn backend.AttnAttrs, pos int, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	rot := cfg.rotaryDim()
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, PosOffset: pos}
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
	if q, err = partialRoPE(ctx, q, cfg.Heads, rot, rope); err != nil {
		return nil, err
	}
	if k, err = partialRoPE(ctx, k, kv, rot, rope); err != nil {
		return nil, err
	}
	if capture != nil {
		capture(k, v)
	}
	a, err := exec3(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return projBias(ctx, a, b.Wdense, b.Bdense)
}

// mlp runs Phi's biased GELU feed-forward on the normalized input xn [seq, dim]:
// fc1 → GELU → fc2. Phi's config hidden_act is "gelu_new" (tanh approximation); GoAI's OpGELU
// is the exact erf GELU, so this leaves a small approximation residual (measured by the
// parity anchor). Mirrors PhiMLP.forward.
func (m *Phi) mlp(ctx *backend.Context, b *PhiBlock, xn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := projBias(ctx, xn, b.Wfc1, b.Bfc1)
	if err != nil {
		return nil, err
	}
	if h, err = exec1a(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return projBias(ctx, h, b.Wfc2, b.Bfc2)
}

// Params returns every trainable tensor for optimizers.
func (m *Phi) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wdense, b.Bq, b.Bk, b.Bv, b.Bdense, b.Wfc1, b.Bfc1, b.Wfc2, b.Bfc2)
		ps = append(ps, b.InputNorm.Params()...)
	}
	ps = append(ps, m.FinalNorm.Params()...)
	ps = append(ps, m.Out, m.OutBias)
	return ps
}
