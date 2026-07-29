package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Nemotron is NVIDIA's Nemotron decoder-only transformer (transformers'
// NemotronForCausalLM). It is a Llama-shaped decoder — RoPE, grouped-query attention, a
// sequential two-norm residual, an untied LM head — but takes THREE departures that make it
// a SEPARATE type (like [StableLM] / [Cohere]) rather than a [Llama] variant:
//
//   - LayerNorm1P (not RMSNorm): input_layernorm, post_attention_layernorm and the final
//     model.norm are NemotronLayerNorm1P — an ordinary mean-centered LayerNorm whose
//     effective gain is (1 + weight). transformers stores the gain as an OFFSET from 1 and
//     applies F.layer_norm(x, …, self.weight + 1.0, self.bias, eps); there IS a bias β.
//     GoAI loads it as a plain nn.LayerNorm with γ = checkpoint_weight + 1 (the +1 folded in
//     at load, exactly like [gemmaRMS] folds Gemma's (1+w) RMSNorm) and β = checkpoint_bias.
//     Getting the +1 fold or the β wrong — or treating the norm as an RMSNorm or as a
//     weight-only LayerNorm — moves the logits far off the golden. See [NemotronFromHF].
//
//   - ReLU² MLP (hidden_act="relu2"), NOT SwiGLU: a plain two-layer MLP with no gate and no
//     bias (mlp_bias=false) — down_proj(relu²(up_proj(x))), where relu²(y) = relu(y)². GoAI
//     computes u = x·Wup; r = ReLU(u); out = (r ⊙ r)·Wdown. See [NemotronBlock.mlp].
//
//   - PARTIAL rotary (partial_rotary_factor, e.g. 0.5): only the first rotaryDim channels of
//     each head are rotated, the rest pass through. rotaryDim = int(head_dim ·
//     partial_rotary_factor). Nemotron's rotate_half is the split-half convention — the same
//     as GoAI's OpRoPE — so [partialRoPE] applies directly with no interleave permutation
//     (unlike Cohere). See [partialRoPE].
//
// The residual is SEQUENTIAL (Llama-style two-norm block, NOT parallel): x = x +
// attention(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x)). There is no
// attention bias (attention_bias=false) and no QK-norm. Attention uses the standard
// 1/√head_dim score scale, supports GQA (num_key_value_heads) and an UNTIED LM head
// (lm_head.weight). Load a Hugging Face NemotronForCausalLM checkpoint with [NemotronFromHF].
type Nemotron struct {
	Config NemotronConfig   // the model hyperparameters
	TokEmb *tensor.Tensor   // [vocab, dim] token embedding (no positional embedding)
	Blocks []*NemotronBlock // the stacked sequential-residual transformer blocks
	Norm   *nn.LayerNorm    // final pre-logits LayerNorm1P (model.norm), γ=w+1 AND bias β
	Out    *tensor.Tensor   // [dim, vocab] output projection (untied lm_head.weightᵀ)
}

// NemotronConfig fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred from
// the checkpoint by [NemotronFromHF]; the remaining fields come from config.json. The rotary
// width is set either directly (RotaryDim) or as a fraction (RotaryPct,
// config.partial_rotary_factor); RotaryDim wins when both are set.
type NemotronConfig struct {
	Vocab     int     // vocabulary size
	Ctx       int     // max context length
	Dim       int     // embedding width (d_model)
	Heads     int     // number of query heads
	KVHeads   int     // key/value heads (GQA); 0 → Heads (standard MHA)
	HeadDim   int     // per-head width (config.head_dim); 0 → Dim/Heads
	Layers    int     // number of transformer blocks
	Hidden    int     // ReLU² MLP inner width (intermediate_size)
	Eps       float64 // LayerNorm epsilon (norm_eps, ~1e-5)
	RopeBase  float64 // RoPE frequency base θ; 0 → 10000
	RotaryDim int     // rotated channels per head; 0 → int(headDim·RotaryPct)
	RotaryPct float64 // partial_rotary_factor; used when RotaryDim==0; 0 → 0.5
}

// NemotronBlock is one sequential-residual Nemotron transformer block. It carries TWO
// LayerNorm1P norms (each with γ=w+1 and a bias β): InputNorm gates the attention sublayer
// and PostAttnNorm gates the ReLU² MLP, each on its own residual add. The MLP is a plain
// up→relu²→down pair (no gate, no bias) rather than a SwiGLU.
type NemotronBlock struct {
	InputNorm      *nn.LayerNorm  // input_layernorm (γ=w+1, bias β), feeds attention
	PostAttnNorm   *nn.LayerNorm  // post_attention_layernorm (γ=w+1, bias β), feeds MLP
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (no bias)
	Wup, Wdown     *tensor.Tensor // ReLU² MLP: [dim,hidden] up, [hidden,dim] down (no bias)
}

func (c NemotronConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// headDim is the per-head width (config.head_dim); Nemotron does not decouple it in its
// small configs, so it defaults to Dim/Heads.
func (c NemotronConfig) headDim() int {
	if c.HeadDim > 0 {
		return c.HeadDim
	}
	return c.Dim / c.Heads
}

// rotaryDim is the number of channels per head that carry RoPE (partial rotary). It is
// RotaryDim when set, else int(head_dim · RotaryPct) with RotaryPct defaulting to 0.5.
func (c NemotronConfig) rotaryDim() int {
	if c.RotaryDim > 0 {
		return c.RotaryDim
	}
	pct := c.RotaryPct
	if pct == 0 {
		pct = 0.5
	}
	return int(float64(c.headDim()) * pct)
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Nemotron) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// embed gathers the token embeddings for the prompt (Nemotron has no positional embedding —
// position enters through RoPE inside attention, and there is no embedding scale).
func (m *Nemotron) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Nemotron prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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

// hidden runs the sequential-residual block stack and the final LayerNorm1P on an embedding x
// [seq, dim], returning the pre-logits hidden states [seq, dim].
func (m *Nemotron) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenCapture(ctx, x, nil)
}

// hiddenCapture is hidden with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-partial-RoPE k and raw post-projection v [seq, kvWidth] — exactly the
// rows a KV-cache decode appends per token ([Nemotron.DecodeStep]), which is
// what lets [Nemotron.Prefill] seed a cache from ONE batched pass over the
// prompt. The tensors handed to capture are the same ones the ongoing forward
// consumes (callers must not mutate them; the cache copies rows into its own
// buffers). A nil capture is the plain forward: no extra ops are recorded, so
// the semantics of hidden are byte-identical to before this hook existed.
func (m *Nemotron) hiddenCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for l, b := range m.Blocks {
		// Sequential residual (Llama-style): input_layernorm → attention → add; then
		// post_attention_layernorm → ReLU² MLP → add.
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
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		xn2, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := b.mlp(ctx, xn2)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	return m.Norm.Forward(ctx, x)
}

// mlp computes Nemotron's ReLU² feed-forward on the normalized input xn [seq, dim]:
// down_proj(relu²(up_proj(x))) with relu²(y) = relu(y)² and no gate, no bias. Mirrors
// NemotronMLP.forward. Every op (matmul/relu/mul) carries a VJP, so the MLP is trainable.
func (b *NemotronBlock) mlp(ctx *backend.Context, xn *tensor.Tensor) (*tensor.Tensor, error) {
	u, err := exec1(ctx, backend.OpMatMul, nil, xn, b.Wup)
	if err != nil {
		return nil, err
	}
	r, err := exec1a(ctx, backend.OpReLU, nil, u)
	if err != nil {
		return nil, err
	}
	r2, err := exec2(ctx, backend.OpMul, nil, r, r)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, r2, b.Wdown)
}

// attention computes Nemotron's multi-head attention over the normalized input xn [seq,
// dim]: project q/k/v, apply PARTIAL split-half RoPE (only the first rotaryDim channels of
// each head rotate), then standard causal GQA via OpMHA and the o_proj. Mirrors
// NemotronAttention.forward (no attention bias, no QK-norm, 1/√head_dim scale). A non-nil
// capture receives the post-partial-RoPE k and raw v right before OpMHA (the Prefill KV
// tap, see [Nemotron.hiddenCapture]); nil is the plain forward.
func (m *Nemotron) attention(ctx *backend.Context, b *NemotronBlock, xn *tensor.Tensor, attn backend.AttnAttrs, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
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
	a, err := exec3(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return project(ctx, a, b.Wo)
}

// Params returns every trainable tensor for optimizers.
func (m *Nemotron) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wo, b.Wup, b.Wdown)
		ps = append(ps, b.InputNorm.Params()...)
		ps = append(ps, b.PostAttnNorm.Params()...)
	}
	ps = append(ps, m.Norm.Params()...)
	ps = append(ps, m.Out)
	return ps
}
