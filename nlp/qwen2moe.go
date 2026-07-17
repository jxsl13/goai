package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Qwen2MoE is Alibaba's Qwen2-MoE sparse-Mixture-of-Experts decoder (transformers'
// Qwen2MoeForCausalLM). It combines Qwen2 attention with a MoE feed-forward that has a
// distinctive extra part — a SHARED expert applied to every token alongside the sparse
// top-k experts:
//
//   - Attention = Qwen2: pre-norm RMSNorm, split-half RoPE, grouped-query attention, and
//     — unlike Mixtral/Llama — a BIAS on the q/k/v projections (o_proj stays bias-free),
//     no QK-norm, standard 1/√head_dim score scale.
//
//   - FFN = sparse MoE + SHARED expert (Qwen2MoeSparseMoeBlock). The sparse part is the
//     Mixtral fused top-k MoE verbatim (a router picks the top-k of N SwiGLU experts, each
//     moe_intermediate_size wide, and mixes their renormalized outputs). Added to it is a
//     single SwiGLU "shared expert" of shared_expert_intermediate_size width, run on ALL
//     tokens and scaled by a sigmoid gate: for every token x the block output is
//
//     y = sparse_moe(x) + sigmoid(x·shared_expert_gateᵀ) · shared_expert(x)
//
//     where shared_expert_gate is a bias-free Linear [dim→1]. This shared expert is the one
//     structural addition over [Mixtral]/Qwen3-MoE; see [Qwen2MoE.ffn].
//
// RMSNorm is the standard (weight-only, no 1+w) variant, the LM head is untied, and RoPE/GQA
// are standard. Load a Hugging Face checkpoint with [Qwen2MoeFromHF].
//
// Limitation (shared with Qwen3-MoE, see [Qwen3MoeFromHF]): the sparse combine always
// RENORMALIZES the top-k router weights (nn.SparseMoE), matching transformers with
// norm_topk_prob=True. A norm_topk_prob=False checkpoint routes to the same experts but
// weights them un-renormalized and so would NOT match this loader.
type Qwen2MoE struct {
	Config Qwen2MoeConfig
	TokEmb *tensor.Tensor   // [vocab, dim] token embedding (model.embed_tokens)
	Blocks []*Qwen2MoeBlock // stacked pre-norm transformer blocks
	Norm   *nn.RMSNorm      // final pre-logits RMSNorm (model.norm)
	Out    *tensor.Tensor   // [dim, vocab] untied output projection (lm_head.weightᵀ)
}

// Qwen2MoeConfig fixes the model geometry. Vocab, Dim, Layers, Hidden, SharedHidden and
// Experts are inferred from the checkpoint by [Qwen2MoeFromHF]; the rest come from config.json.
type Qwen2MoeConfig struct {
	Vocab        int     // vocabulary size
	Ctx          int     // max context length (max_position_embeddings)
	Dim          int     // embedding width (hidden_size)
	Heads        int     // query heads (num_attention_heads)
	KVHeads      int     // key/value heads (GQA, num_key_value_heads); 0 → Heads
	Layers       int     // number of transformer blocks
	Hidden       int     // per-expert SwiGLU inner width (moe_intermediate_size)
	SharedHidden int     // shared-expert SwiGLU inner width (shared_expert_intermediate_size)
	Experts      int     // number of routed experts (num_experts)
	TopK         int     // experts per token (num_experts_per_tok); 0 → 2
	Eps          float64 // RMSNorm epsilon (rms_norm_eps)
	RopeBase     float64 // RoPE frequency base θ (rope_theta); 0 → 10000
}

// Qwen2MoeBlock is one pre-norm Qwen2-MoE block: Qwen2 biased attention + a
// sparse-MoE-plus-shared-expert FFN.
type Qwen2MoeBlock struct {
	AttnNorm       *nn.RMSNorm    // input_layernorm, feeds attention
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (o_proj bias-free)
	Bq, Bk, Bv     *tensor.Tensor // q/k/v projection biases (Qwen2 use_bias on QKV only)
	FFNNorm        *nn.RMSNorm    // post_attention_layernorm, feeds the FFN
	MoE            *nn.SparseMoE  // sparse top-k routed experts (Mixtral fused layout)
	Shared         *nn.SwiGLU     // shared expert applied to every token
	SharedGate     *tensor.Tensor // [dim, 1] bias-free gate; sigmoid scales the shared expert
}

func (c Qwen2MoeConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Qwen2MoE) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	return m.forwardCapture(ctx, tokens, nil, false)
}

// forwardCapture is Forward with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-bias, POST-RoPE k and post-bias v [seq, kvWidth] — exactly the rows
// [Qwen2MoE.DecodeStep] appends per token, which is what lets [Qwen2MoE.Prefill]
// seed a cache from ONE batched pass over the prompt. The tensors handed to
// capture are the same ones the ongoing forward consumes (callers must not
// mutate them; the cache copies rows into its own buffers). A nil capture with
// sparseFFN=false is the plain forward: no extra ops are recorded, so
// tape/training semantics of Forward are byte-identical to before this hook
// existed.
//
// sparseFFN is forwarded to [Qwen2MoE.ffn]'s decode flag and selects the ROUTED
// experts' evaluation path: false is the dense MoE.Forward (training/tape),
// true is the same inference-only multi-row MoE.ForwardDecode DecodeStep uses
// (the shared expert runs identically either way). Mathematically identical,
// but NOT bit-identical: the dense OpMoECombine kernel accumulates
// acc += (w/denom)·e in one fused expression, which Go may compile to an FMA
// (spec-sanctioned fusion, arm64 FMADD), while the sparse path rounds the
// product through a stored OpMul tensor before the OpAdd — a ~1-ulp divergence.
// Prefill therefore passes sparseFFN=true so its residual stream (and thus
// every later layer's cached k/v) matches DecodeStep exactly, bit for bit.
func (m *Qwen2MoE) forwardCapture(ctx *backend.Context, tokens []int, capture func(layer int, k, v *tensor.Tensor), sparseFFN bool) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Qwen2MoE prompt length %d outside (0,%d]", seq, m.Config.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
	if err != nil {
		return nil, err
	}

	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for l, b := range m.Blocks {
		// attention sublayer (Qwen2: biased q/k/v, bias-free o_proj)
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := projBias(ctx, xb, b.Wq, b.Bq)
		if err != nil {
			return nil, err
		}
		k, err := projBias(ctx, xb, b.Wk, b.Bk)
		if err != nil {
			return nil, err
		}
		v, err := projBias(ctx, xb, b.Wv, b.Bv)
		if err != nil {
			return nil, err
		}
		if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, k); err != nil {
			return nil, err
		}
		if capture != nil {
			capture(l, k, v)
		}
		a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		o, err := project(ctx, a, b.Wo)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// sparse-MoE + shared-expert FFN sublayer
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := m.ffn(ctx, b, xf, sparseFFN)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// ffn is Qwen2-MoE's Qwen2MoeSparseMoeBlock on the normalized input xf [seq, dim]: the sparse
// top-k routed MoE output PLUS the sigmoid-gated shared expert, added token-wise. Concretely
//
//	sparse = MoE.Forward(xf)                       // [seq, dim]
//	shared = Shared.Forward(xf)                     // [seq, dim] SwiGLU over ALL tokens
//	g      = sigmoid(xf · SharedGate)               // [seq, 1]  gate logits → probability
//	out    = sparse + shared ⊙ g                    // g broadcasts [seq,1] over [seq,dim]
//
// The [seq,1] gate broadcasts across the hidden dim through OpMul (row-broadcast is a
// backend fast path with a matching VJP, §V2), so both the sparse and shared paths stay
// differentiable — the shared expert, its gate, and every routed expert receive gradients.
func (m *Qwen2MoE) ffn(ctx *backend.Context, b *Qwen2MoeBlock, xf *tensor.Tensor, decode bool) (*tensor.Tensor, error) {
	// The shared expert always runs on every token; only the ROUTED experts differ between
	// dense (training/prefill) and sparse-decode (evaluate just the top-k) — identical output.
	var sparse *tensor.Tensor
	var err error
	if decode {
		sparse, _, err = b.MoE.ForwardDecode(ctx, xf)
	} else {
		sparse, _, err = b.MoE.Forward(ctx, xf)
	}
	if err != nil {
		return nil, err
	}
	shared, err := b.Shared.Forward(ctx, xf)
	if err != nil {
		return nil, err
	}
	gl, err := exec1(ctx, backend.OpMatMul, nil, xf, b.SharedGate) // [seq, 1]
	if err != nil {
		return nil, err
	}
	g, err := exec1(ctx, backend.OpSigmoid, nil, gl)
	if err != nil {
		return nil, err
	}
	sg, err := exec1(ctx, backend.OpMul, nil, shared, g) // [seq,dim] ⊙ [seq,1] → [seq,dim]
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpAdd, nil, sparse, sg)
}

// Params returns every trainable tensor for optimizers: embeddings, per-block attention
// (weights + q/k/v biases), the FFN norm, the router, every routed expert, the shared
// expert and its gate, plus the final norm and the LM head.
func (m *Qwen2MoE) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo, b.Bq, b.Bk, b.Bv,
			b.FFNNorm.Gamma, b.MoE.Router.W, b.SharedGate)
		for _, e := range b.MoE.Experts {
			ps = append(ps, e.Wgate, e.Wup, e.Wdown)
		}
		ps = append(ps, b.Shared.Params()...)
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
