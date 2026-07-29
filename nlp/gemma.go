package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Gemma is Google's Gemma (v1) decoder-only transformer (Gemma Team 2024). It is
// close to [Llama] — pre-norm RMSNorm, RoPE, grouped-query attention, a gated FFN,
// no attention bias — but differs in four faithful details this type captures:
//
//   - the token embeddings are scaled by √dim ("normalizer") right after lookup;
//   - RMSNorm uses (1 + weight) rather than weight (folded in at load: γ ← γ+1);
//   - the FFN is GeGLU (GELU gate) instead of SwiGLU (SiLU gate);
//   - head_dim is decoupled from dim/heads (dim need NOT equal heads·head_dim), so
//     the q/k/v/o projections carry their own inner width.
//
// The LM head is tied to the embedding. Load a Hugging Face checkpoint with
// [GemmaFromHF]. Real Gemma uses the tanh-approx GELU (gelu_pytorch_tanh); GoAI's
// OpGELU is the exact erf GELU, so loaded logits carry a small approximation
// residual (as with T5 v1.1), documented and bounded by the parity test.
type Gemma struct {
	Config    GemmaConfig    // model geometry: width, heads, head_dim, layers (see GemmaConfig)
	TokEmb    *tensor.Tensor // [vocab, dim] tied token embedding
	Blocks    []*GemmaBlock  // the stacked pre-norm Gemma blocks
	FinalNorm *nn.RMSNorm    // final pre-logits RMSNorm (model.norm)

	outT *tensor.Tensor // cached [dim, vocab] tied-head transpose (see tiedHead)
}

// tiedHead returns the [dim, vocab] transpose of the tied token embedding used as the
// LM head, computed once and cached — re-transposing the whole [vocab, dim] table on
// EVERY Forward and decode step cost ~220ms per call at Llama-scale dims (§V22). If
// TokEmb is mutated in place (fine-tuning), call [Gemma.RefreshTiedHead] afterwards so
// the head reflects the updated embedding.
func (m *Gemma) tiedHead() *tensor.Tensor {
	if m.outT == nil {
		m.outT = transpose2D(m.TokEmb)
	}
	return m.outT
}

// RefreshTiedHead drops the cached tied-head transpose so the next Forward/DecodeStep
// recomputes it from the current TokEmb values. Call after optimizer steps that update
// the token embedding (the cache otherwise serves the pre-update head).
func (m *Gemma) RefreshTiedHead() { m.outT = nil }

// GemmaConfig fixes the model geometry. Dim, Vocab, Layers, FFN and HeadDim are
// inferred from the checkpoint by GemmaFromHF; Heads/KVHeads/Eps/RopeBase/Ctx come
// from config.json.
type GemmaConfig struct {
	Vocab    int     // vocabulary size
	Ctx      int     // maximum context length in tokens
	Dim      int     // d_model (embedding width)
	Heads    int     // query heads
	KVHeads  int     // key/value heads (GQA); 0 → Heads
	HeadDim  int     // per-head width (may differ from Dim/Heads)
	Layers   int     // number of decoder layers
	FFN      int     // GeGLU inner width
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE θ; 0 → 10000
}

// GemmaBlock is one pre-norm Gemma block. The RMSNorm gains already carry the +1
// offset (folded at load), so nn.RMSNorm applies Gemma's (1+w) directly.
type GemmaBlock struct {
	AttnNorm       *nn.RMSNorm    // RMSNorm before attention (input_layernorm)
	Wq, Wk, Wv, Wo *tensor.Tensor // [dim, heads·head_dim] … [heads·head_dim, dim]
	FFNNorm        *nn.RMSNorm    // RMSNorm before the FFN (post_attention_layernorm)
	FFN            *nn.GLU        // GeGLU
}

func (c GemmaConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Gemma) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	return m.forwardCapture(ctx, tokens, nil)
}

// forwardCapture is Forward with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// POST-RoPE k and raw v [seq, kvWidth] — exactly the rows [Gemma.DecodeStep]
// appends per token, which is what lets [Gemma.Prefill] seed a cache from ONE
// batched pass over the prompt. The √dim embedding "normalizer" is applied
// before the blocks exactly as in DecodeStep's embedOne, so the residual stream
// feeding each projection is identical. The tensors handed to capture are the
// same ones the ongoing forward consumes (callers must not mutate them; the
// cache copies rows into its own buffers). A nil capture is the plain forward:
// no extra ops are recorded, so tape/training semantics of Forward are
// byte-identical to before this hook existed.
func (m *Gemma) forwardCapture(ctx *backend.Context, tokens []int, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Gemma prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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
	// Gemma scales the embeddings by √dim before the blocks (the "normalizer"). The
	// tied LM head deliberately uses the UNSCALED embedding, so this cannot be folded
	// into the table — it is a runtime scalar broadcast on the residual stream only.
	scale := tensor.New(tensor.F64, tensor.Shape{})
	scale.Storage().F64()[0] = math.Sqrt(float64(m.Config.Dim))
	if x, err = exec2(ctx, backend.OpMul, nil, x, scale); err != nil {
		return nil, err
	}

	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv})
	for l, b := range m.Blocks {
		// attention sublayer
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
		if x, err = exec2(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// FFN sublayer (GeGLU)
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Tied LM head: logits = hidden · embedᵀ.
	return exec1(ctx, backend.OpMatMul, nil, x, m.tiedHead())
}

// Params returns every trainable tensor for optimizers.
func (m *Gemma) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb, m.FinalNorm.Gamma}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo, b.FFNNorm.Gamma)
		ps = append(ps, b.FFN.Params()...)
	}
	return ps
}
