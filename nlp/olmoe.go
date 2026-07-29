package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// OLMoE is Allen AI's OLMoE sparse-Mixture-of-Experts decoder (transformers'
// OlmoeForCausalLM). It is Llama-shaped — RMSNorm, rotary position embeddings (RoPE),
// grouped-query attention, no biases — with two distinguishing features that make it a
// SEPARATE type rather than a [Mixtral] or [Llama] variant:
//
//   - full-width QK-norm: an RMSNorm is applied to the ENTIRE q_proj / k_proj output
//     (over num_heads·head_dim and num_kv_heads·head_dim respectively) BEFORE the split
//     into heads and RoPE — q = q_norm(Wq·x), k = k_norm(Wk·x). Because the norm spans
//     the full projection width (not per-head like Qwen3-MoE), it is a plain
//     [nn.RMSNorm.Forward] on the [seq, heads·head_dim] tensor, exactly as in [OLMo2].
//     The gains are standard RMSNorm (γ·x̂, NOT Gemma's (1+γ) variant).
//
//   - sparse-MoE FFN: each block's feed-forward is a top-k Mixture-of-Experts (a router
//     picks the top-k of N SwiGLU experts per token and mixes them, nn.SparseMoE) — the
//     same fused-expert layout as [Mixtral], with NO shared expert.
//
// Unlike [OLMo2] (which is post-norm), OLMoE uses the STANDARD Llama pre-norm sequential
// residual: x = x + attn(input_layernorm(x)); x = x + moe(post_attention_layernorm(x)).
// RoPE is full split-half, GQA is standard, there are no attention biases, and the LM
// head is untied by default (tie_word_embeddings=false). A final model.norm precedes the
// head. Load a Hugging Face OlmoeForCausalLM checkpoint with [OLMoEFromHF].
type OLMoE struct {
	Config OLMoEConfig    // the model hyperparameters
	TokEmb *tensor.Tensor // [vocab, dim] token embedding (no positional embedding)
	Blocks []*OLMoEBlock  // the stacked pre-norm sparse-MoE transformer blocks
	Norm   *nn.RMSNorm    // final pre-logits RMSNorm (model.norm)
	Out    *tensor.Tensor // [dim, vocab] output projection (untied by default)
}

// OLMoEConfig fixes the model geometry. Dim, Vocab, Layers, Hidden and Experts are
// inferred from the checkpoint by [OLMoEFromHF]; the remaining fields come from
// config.json. HeadDim is decoupled (config.head_dim) and inferred from q_proj when 0.
type OLMoEConfig struct {
	Vocab    int     // vocabulary size
	Ctx      int     // max context length
	Dim      int     // embedding width (d_model)
	Heads    int     // number of query heads
	KVHeads  int     // key/value heads (GQA); 0 → Heads (standard MHA)
	HeadDim  int     // per-head width (may differ from Dim/Heads); 0 → Dim/Heads
	Layers   int     // number of transformer blocks
	Hidden   int     // per-expert SwiGLU inner width (intermediate_size)
	Experts  int     // number of experts (num_experts)
	TopK     int     // experts per token (num_experts_per_tok); 0 → 2
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE frequency base θ; 0 → 10000
}

// OLMoEBlock is one pre-norm OLMoE transformer block: full-width-QK-norm attention + a
// sparse-MoE FFN. AttnNorm (input_layernorm) and FFNNorm (post_attention_layernorm)
// normalize the residual stream before each sublayer (standard Llama pre-norm). QNorm and
// KNorm are full-width RMSNorms over the whole q/k projections, applied before RoPE.
type OLMoEBlock struct {
	AttnNorm       *nn.RMSNorm    // input_layernorm (before attention)
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (no bias); Wk/Wv are [dim, KVHeads·headDim]
	QNorm, KNorm   *nn.RMSNorm    // full-width q/k RMSNorm (over heads·headDim / kv·headDim) applied before RoPE
	FFNNorm        *nn.RMSNorm    // post_attention_layernorm (before the MoE FFN)
	MoE            *nn.SparseMoE  // sparse top-k Mixture-of-Experts feed-forward
}

func (c OLMoEConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *OLMoE) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// embed gathers the token embeddings for the prompt (OLMoE has no positional embedding —
// position enters through RoPE inside attention, and there is no embedding scale).
func (m *OLMoE) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: OLMoE prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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

// hidden runs the pre-norm sparse-MoE block stack and the final RMSNorm on an embedding x
// [seq, dim], returning the pre-logits hidden states [seq, dim].
func (m *OLMoE) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenCapture(ctx, x, nil, false)
}

// hiddenCapture is hidden with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-k_norm, POST-RoPE k and raw v [seq, kvWidth] — exactly the rows
// [OLMoE.DecodeStep] appends per token, which is what lets [OLMoE.Prefill] seed
// a cache from ONE batched pass over the prompt. The tensors handed to capture
// are the same ones the ongoing forward consumes (callers must not mutate them;
// the cache copies rows into its own buffers). A nil capture with
// sparseFFN=false is the plain forward: no extra ops are recorded, so
// tape/training semantics of hidden are byte-identical to before this hook
// existed.
//
// sparseFFN selects the MoE evaluation path: false is the dense b.MoE.Forward
// (training/tape; every expert evaluated), true is the same inference-only
// multi-row b.MoE.ForwardDecode DecodeStep uses. The two are mathematically
// identical (same renormalized top-k routing), but NOT bit-identical: the dense
// OpMoECombine kernel accumulates acc += (w/denom)·e in one fused expression,
// which Go may compile to an FMA (spec-sanctioned fusion, arm64 FMADD), while
// the sparse path rounds the product through a stored OpMul tensor before the
// OpAdd — a ~1-ulp divergence (§B64, first seen on Mixtral). Prefill therefore
// passes sparseFFN=true so its residual stream (and thus every later layer's
// cached k/v) matches DecodeStep exactly, bit for bit.
func (m *OLMoE) hiddenCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor), sparseFFN bool) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for l, b := range m.Blocks {
		// Attention sublayer (pre-norm): x = x + attn(input_layernorm(x)).
		var tap func(k, v *tensor.Tensor)
		if capture != nil {
			tap = func(k, v *tensor.Tensor) { capture(l, k, v) }
		}
		a, err := m.attention(ctx, b, x, attn, tap)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// sparse-MoE FFN sublayer (pre-norm): x = x + moe(post_attention_layernorm(x))
		// (dense for training, ForwardDecode for prefill — see the sparseFFN contract above).
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if sparseFFN {
			ff, _, err = b.MoE.ForwardDecode(ctx, xf)
		} else {
			ff, _, err = b.MoE.Forward(ctx, xf)
		}
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	return m.Norm.Forward(ctx, x)
}

// attention computes OLMoE's multi-head attention over the pre-normed residual: normalize
// x with input_layernorm, project q/k/v, apply the full-width q_norm/k_norm to the WHOLE
// q/k projections (before the head split), RoPE, then standard causal GQA via OpMHA and
// the o_proj. Mirrors OlmoeAttention.forward exactly. capture, when non-nil, receives the
// post-k_norm POST-RoPE k and the raw v — the per-token cache rows — for [OLMoE.Prefill].
func (m *OLMoE) attention(ctx *backend.Context, b *OLMoEBlock, x *tensor.Tensor, attn backend.AttnAttrs, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	xb, err := b.AttnNorm.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	q, err := project(ctx, xb, b.Wq)
	if err != nil {
		return nil, err
	}
	k, err := project(ctx, xb, b.Wk)
	if err != nil {
		return nil, err
	}
	v, err := project(ctx, xb, b.Wv)
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

// Params returns every trainable tensor for optimizers (router + all experts included).
func (m *OLMoE) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo)
		ps = append(ps, b.QNorm.Gamma, b.KNorm.Gamma, b.FFNNorm.Gamma, b.MoE.Router.W)
		for _, e := range b.MoE.Experts {
			ps = append(ps, e.Wgate, e.Wup, e.Wdown)
		}
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
