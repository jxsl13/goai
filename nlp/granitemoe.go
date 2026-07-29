package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GraniteMoE is IBM's Granite sparse-Mixture-of-Experts decoder (transformers'
// GraniteMoeForCausalLM). It is the MoE sibling of the dense [Llama]-plus-Granite-scalars
// path (see [GraniteFromHF]): the attention is plain Llama-style — RMSNorm, RoPE,
// grouped-query attention, no biases, no QK-norm — and each block's feed-forward is a
// sparse top-k MoE (a router picks the top-k of N SwiGLU experts and mixes their outputs
// with the renormalized router weights, nn.SparseMoE), exactly like [Mixtral]. On top of
// that structure Granite layers four learned-at-config scalar multipliers:
//
//   - embedding_multiplier: inputs_embeds *= EmbeddingMult after the token lookup,
//   - attention_multiplier: the pre-softmax attention scale is AttentionMult
//     (replacing the default 1/√headDim),
//   - residual_multiplier: every residual add scales the sublayer output — BOTH the
//     attention output AND the MoE-FFN output — by ResidualMult, and
//   - logits_scaling: the final logits are divided by LogitsScale.
//
// Real GraniteMoE checkpoints set these ≠ 1; each is composed with the sparse MoE below.
// Load a Hugging Face checkpoint with [GraniteMoeFromHF].
type GraniteMoE struct {
	Config GraniteMoeConfig   // geometry and Granite scalar multipliers (see GraniteMoeConfig)
	TokEmb *tensor.Tensor     // [vocab, dim] token embedding
	Blocks []*GraniteMoeBlock // the attention + sparse-MoE blocks
	Norm   *nn.RMSNorm        // final pre-logits RMSNorm
	Out    *tensor.Tensor     // [dim, vocab] output projection (untied)
}

// GraniteMoeConfig fixes the model geometry. Dim, Vocab, Layers, Hidden and Experts are
// inferred from the checkpoint by [GraniteMoeFromHF]; the rest come from config.json,
// including Granite's four scalar multipliers.
type GraniteMoeConfig struct {
	Vocab    int     // vocabulary size
	Ctx      int     // maximum context length in tokens
	Dim      int     // embedding width (hidden_size)
	Heads    int     // query heads (num_attention_heads)
	KVHeads  int     // GQA; 0 → Heads
	Layers   int     // number of decoder layers
	Hidden   int     // per-expert SwiGLU inner width (intermediate_size)
	Experts  int     // number of experts (num_local_experts)
	TopK     int     // experts per token (num_experts_per_tok); 0 → 2
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE θ; 0 → 10000

	// Granite scalar multipliers (transformers GraniteMoeConfig). Real checkpoints set
	// these ≠ 1; a 0 ("unset") or identity (1) makes the corresponding hook a byte-exact
	// no-op, so a plain-Mixtral-style config degrades gracefully.
	EmbeddingMult float64 // inputs_embeds *= EmbeddingMult after lookup; 0 or 1 → identity
	AttentionMult float64 // pre-softmax attention scale = AttentionMult (replaces 1/√headDim); 0 → default
	ResidualMult  float64 // each residual add is x = residual + sublayer·ResidualMult (attn AND MoE); 0 or 1 → identity
	LogitsScale   float64 // final logits are divided by LogitsScale; 0 or 1 → identity
}

// GraniteMoeBlock is one pre-norm GraniteMoE block: Llama-style attention (no bias, no
// QK-norm) followed by a sparse-MoE FFN. The Granite scalars live in the enclosing
// [GraniteMoeConfig] and are applied by Forward/DecodeStep, not stored per-block.
type GraniteMoeBlock struct {
	AttnNorm       *nn.RMSNorm    // RMSNorm before attention (input_layernorm)
	Wq, Wk, Wv, Wo *tensor.Tensor // bias-free attention projections (q/k/v/o)
	FFNNorm        *nn.RMSNorm    // RMSNorm before the MoE FFN (post_attention_layernorm)
	MoE            *nn.SparseMoE  // sparse top-k SwiGLU expert bank
}

func (c GraniteMoeConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// attnScale returns the AttnAttrs.Scale for the Granite attention_multiplier. OpMHA
// builds in a 1/√headDim, and AttnAttrs.Scale is an EXTRA factor on top of it, so
// Scale = AttentionMult·√headDim leaves a net pre-softmax scale of AttentionMult
// (matching transformers' GraniteMoeAttention scaling = attention_multiplier). A 0
// AttentionMult leaves Scale 0, i.e. the default 1/√headDim.
func (c GraniteMoeConfig) attnScale() float64 {
	if c.AttentionMult == 0 {
		return 0
	}
	return c.AttentionMult * math.Sqrt(float64(c.Dim/c.Heads))
}

// Forward computes logits [seq, vocab] for the prompt tokens, applying the four Granite
// scalar multipliers around a Mixtral-style attention + sparse-MoE stack.
func (m *GraniteMoE) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	return m.forwardCapture(ctx, tokens, nil, false)
}

// forwardCapture is Forward with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// POST-RoPE k and raw v [seq, kvWidth] — exactly the rows
// [GraniteMoE.DecodeStep] appends per token, which is what lets
// [GraniteMoE.Prefill] seed a cache from ONE batched pass over the prompt. All
// four Granite scalars thread through unchanged (embedding, attention, residual,
// logits); they scale the residual stream, never the cached k/v rows themselves,
// beyond their effect on the layer inputs — identical in both paths. The tensors
// handed to capture are the same ones the ongoing forward consumes (callers must
// not mutate them; the cache copies rows into its own buffers). A nil capture
// with sparseFFN=false is the plain forward: no extra ops are recorded, so
// tape/training semantics of Forward are byte-identical to before this hook
// existed.
//
// sparseFFN selects the MoE evaluation path: false is the dense b.MoE.Forward
// (training/tape), true is the same inference-only multi-row b.MoE.ForwardDecode
// DecodeStep uses. Mathematically identical, but NOT bit-identical: the dense
// OpMoECombine kernel accumulates acc += (w/denom)·e in one fused expression,
// which Go may compile to an FMA (spec-sanctioned fusion, arm64 FMADD), while
// the sparse path rounds the product through a stored OpMul tensor before the
// OpAdd — a ~1-ulp divergence. Prefill therefore passes sparseFFN=true so its
// residual stream (and thus every later layer's cached k/v) matches DecodeStep
// exactly, bit for bit.
func (m *GraniteMoE) forwardCapture(ctx *backend.Context, tokens []int, capture func(layer int, k, v *tensor.Tensor), sparseFFN bool) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: GraniteMoE prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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
	// Granite: inputs_embeds *= embedding_multiplier.
	if x, err = scaleScalar(ctx, x, m.Config.EmbeddingMult); err != nil {
		return nil, err
	}

	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true, Scale: cfg.attnScale()}
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv})
	for l, b := range m.Blocks {
		// attention sublayer (Llama-style, bias-free, no QK-norm)
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wv)
		if err != nil {
			return nil, err
		}
		if q, err = exec1a(ctx, backend.OpRoPE, qRoPE, q); err != nil {
			return nil, err
		}
		if k, err = exec1a(ctx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		if capture != nil {
			capture(l, k, v)
		}
		a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		o, err := exec1(ctx, backend.OpMatMul, nil, a, b.Wo)
		if err != nil {
			return nil, err
		}
		// Granite: x = x + o·residual_multiplier.
		if o, err = scaleScalar(ctx, o, cfg.ResidualMult); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// sparse-MoE FFN sublayer (dense for training, ForwardDecode for prefill —
		// see the sparseFFN contract in the method comment)
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
		// Granite: x = x + ff·residual_multiplier (same scalar as the attention residual).
		if ff, err = scaleScalar(ctx, ff, cfg.ResidualMult); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.Out)
	if err != nil {
		return nil, err
	}
	// Granite: logits /= logits_scaling.
	return divLogits(ctx, logits, cfg.LogitsScale)
}

// Params returns every trainable tensor for optimizers (router + all experts included).
func (m *GraniteMoE) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo, b.FFNNorm.Gamma, b.MoE.Router.W)
		for _, e := range b.MoE.Experts {
			ps = append(ps, e.Wgate, e.Wup, e.Wdown)
		}
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
