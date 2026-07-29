package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GPTNeoX is EleutherAI's GPT-NeoX / Pythia decoder-only transformer (transformers'
// GPTNeoXForCausalLM). It is a separate type — rather than a [Llama] variant — because it
// takes FIVE departures from the Llama block, all confirmed against
// transformers.models.gpt_neox.modeling_gpt_neox:
//
//   - PARALLEL residual (use_parallel_residual=True, the default). Unlike sequential
//     transformers, and unlike [Cohere] which shares ONE norm, GPT-NeoX reads the SAME
//     residual stream through TWO separate norms — input_layernorm feeds attention,
//     post_attention_layernorm feeds the MLP — and SUMS both sublayer outputs onto the raw
//     residual (GPTNeoXLayer.forward):
//
//     x = x + attention(input_layernorm(x)) + mlp(post_attention_layernorm(x))
//
//   - Full LayerNorm WITH bias (not RMSNorm, not weight-only). Every norm is a standard
//     nn.LayerNorm carrying both γ and β loaded from the checkpoint — the block's
//     input_layernorm / post_attention_layernorm and the model's final_layer_norm.
//
//   - PACKED, per-head-interleaved QKV with bias. attention.query_key_value.weight is
//     [3·hidden, hidden] and .bias is [3·hidden], but the packing is NOT [all-q; all-k;
//     all-v]. transformers views the output as [num_heads, 3·head_size] and chunks the last
//     axis into q/k/v, so within EACH head's 3·head_size block the first head_size rows are
//     that head's query, the next head_size its key, the last head_size its value.
//     [GPTNeoXFromHF] de-interleaves this into per-projection Wq/Wk/Wv (+ biases).
//
//   - PARTIAL rotary. Only rotary_ndims = int(head_dim · partial_rotary_factor) of each
//     head's channels are rotated; the rest pass through unrotated. GPT-NeoX's rotate_half
//     is split-half (the same convention as GoAI's OpRoPE), so [partialRoPE] applies
//     directly with no interleave permutation (contrast [Cohere]).
//
//   - GELU MLP with bias (not SwiGLU): dense_h_to_4h → GELU → dense_4h_to_h, both Linear
//     layers carrying a bias. hidden_act defaults to "gelu" (exact erf), matching GoAI's
//     OpGELU.
//
// Attention uses the standard 1/√head_dim score scale, MHA (no GQA), and an UNTIED LM head
// (embed_out.weight). Load a Hugging Face GPTNeoXForCausalLM / Pythia checkpoint with
// [GPTNeoXFromHF].
type GPTNeoX struct {
	Config    GPTNeoXConfig   // the model hyperparameters
	TokEmb    *tensor.Tensor  // [vocab, dim] token embedding (gpt_neox.embed_in)
	Blocks    []*GPTNeoXBlock // the stacked parallel-residual transformer blocks
	FinalNorm *nn.LayerNorm   // gpt_neox.final_layer_norm (LayerNorm with bias)
	Out       *tensor.Tensor  // [dim, vocab] output projection (embed_out, untied)
}

// GPTNeoXConfig fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred from
// the checkpoint by [GPTNeoXFromHF]; the remaining fields come from config.json. RotaryDim
// takes precedence when set; otherwise it is derived from RotaryPct · head_dim.
type GPTNeoXConfig struct {
	Vocab     int     // vocabulary size
	Ctx       int     // max context length (max_position_embeddings)
	Dim       int     // embedding width (hidden_size)
	Heads     int     // number of attention heads (GPT-NeoX is MHA: KVHeads == Heads)
	KVHeads   int     // key/value heads; GPT-NeoX has no GQA so this equals Heads (0 → Heads)
	Layers    int     // number of transformer blocks
	Hidden    int     // MLP inner width (intermediate_size)
	Eps       float64 // LayerNorm epsilon (layer_norm_eps, ~1e-5)
	RopeBase  float64 // RoPE frequency base θ (rope_theta); 0 → 10000
	RotaryPct float64 // fraction of head_dim rotated (partial_rotary_factor / rotary_pct); 0 → 1
	RotaryDim int     // explicit rotated-channel count; 0 → int(RotaryPct · head_dim)
}

// GPTNeoXBlock is one parallel-residual GPT-NeoX transformer block. InputNorm feeds the
// attention sublayer and PostAttnNorm feeds the MLP, but BOTH read the same raw residual and
// both outputs are summed onto it. Every projection carries a bias.
type GPTNeoXBlock struct {
	InputNorm, PostAttnNorm *nn.LayerNorm  // input_layernorm / post_attention_layernorm (with bias)
	Wq, Wk, Wv, Wo          *tensor.Tensor // attention projections (query_key_value split + dense)
	Bq, Bk, Bv, Bo          *tensor.Tensor // their biases
	Wh, Bh                  *tensor.Tensor // mlp.dense_h_to_4h weight and bias
	Wout, Bout              *tensor.Tensor // mlp.dense_4h_to_h weight and bias
}

func (c GPTNeoXConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// headDim returns the per-head width (GPT-NeoX uses head_dim = Dim/Heads).
func (c GPTNeoXConfig) headDim() int {
	return c.Dim / c.Heads
}

// rotaryDim returns the number of rotated channels per head. RotaryDim wins when set;
// otherwise int(RotaryPct · head_dim); a zero RotaryPct means full rotary (head_dim).
func (c GPTNeoXConfig) rotaryDim() int {
	if c.RotaryDim > 0 {
		return c.RotaryDim
	}
	hd := c.headDim()
	if c.RotaryPct <= 0 {
		return hd
	}
	return int(float64(hd) * c.RotaryPct)
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *GPTNeoX) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// embed gathers the token embeddings for the prompt (GPT-NeoX has no positional embedding
// and no embedding scale — position enters through the partial RoPE inside attention).
func (m *GPTNeoX) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: GPTNeoX prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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
func (m *GPTNeoX) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenCapture(ctx, x, nil)
}

// hiddenCapture is hidden with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-bias, post-partial-RoPE k and post-bias v [seq, kvWidth] — exactly the
// rows a KV-cache decode appends per token ([GPTNeoX.DecodeStep]), which is
// what lets [GPTNeoX.Prefill] seed a cache from ONE batched pass over the
// prompt. The tensors handed to capture are the same ones the ongoing forward
// consumes (callers must not mutate them; the cache copies rows into its own
// buffers). A nil capture is the plain forward: no extra ops are recorded, so
// the semantics of hidden are byte-identical to before this hook existed.
func (m *GPTNeoX) hiddenCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: cfg.kvHeads(), Causal: true}
	for l, b := range m.Blocks {
		// Parallel residual: input_layernorm feeds attention, post_attention_layernorm feeds
		// the MLP, and both outputs are summed onto the SAME raw residual x.
		an, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var tap func(k, v *tensor.Tensor)
		if capture != nil {
			tap = func(k, v *tensor.Tensor) { capture(l, k, v) }
		}
		a, err := m.attention(ctx, b, an, attn, 0, tap)
		if err != nil {
			return nil, err
		}
		fn, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, fn)
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

// attention computes GPT-NeoX multi-head attention over the normalized input xn [seq, dim]:
// project the (already de-interleaved) q/k/v with their biases, apply the PARTIAL split-half
// RoPE at position offset pos, then standard causal MHA and the biased output dense. Mirrors
// GPTNeoXAttention.forward. A non-nil capture receives the post-partial-RoPE k and post-bias
// v right before OpMHA (the Prefill KV tap, see [GPTNeoX.hiddenCapture]); nil is the plain
// forward.
func (m *GPTNeoX) attention(ctx *backend.Context, b *GPTNeoXBlock, xn *tensor.Tensor, attn backend.AttnAttrs, pos int, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
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
	return projBias(ctx, a, b.Wo, b.Bo)
}

// mlp runs GPT-NeoX's biased GELU feed-forward on the normalized input fn [seq, dim]:
// dense_h_to_4h → exact-erf GELU → dense_4h_to_h. Mirrors GPTNeoXMLP.forward.
func (m *GPTNeoX) mlp(ctx *backend.Context, b *GPTNeoXBlock, fn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := projBias(ctx, fn, b.Wh, b.Bh)
	if err != nil {
		return nil, err
	}
	if h, err = exec1a(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return projBias(ctx, h, b.Wout, b.Bout)
}

// projBias computes x·W + bias (a biased linear layer); bias is always present in GPT-NeoX.
func projBias(ctx *backend.Context, x, w, bias *tensor.Tensor) (*tensor.Tensor, error) {
	y, err := project(ctx, x, w)
	if err != nil {
		return nil, err
	}
	return addBiasIf(ctx, y, bias)
}

// Params returns every trainable tensor for optimizers.
func (m *GPTNeoX) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wo, b.Bq, b.Bk, b.Bv, b.Bo, b.Wh, b.Bh, b.Wout, b.Bout)
		ps = append(ps, b.InputNorm.Params()...)
		ps = append(ps, b.PostAttnNorm.Params()...)
	}
	ps = append(ps, m.FinalNorm.Params()...)
	ps = append(ps, m.Out)
	return ps
}
