package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Gemma2 is Google's Gemma 2 decoder-only transformer (Gemma Team, arXiv:2408.00118).
// It keeps every Gemma (v1) trait — √dim embedding scale at runtime, (1+w) RMSNorm,
// RoPE, grouped-query attention, a GeGLU FFN, decoupled head_dim, and a tied LM head
// (see [Gemma]) — and adds the four faithful details this type captures:
//
//   - sandwich norms: each block has FOUR (1+w) RMSNorms instead of two —
//     input_layernorm → attn → post_attention_layernorm → +residual, then
//     pre_feedforward_layernorm → GeGLU → post_feedforward_layernorm → +residual;
//   - attention-logit soft-cap: the pre-softmax scores are passed through
//     y=c·tanh(x/c) (c = AttnLogitCap) BEFORE the causal mask, smoothly bounding
//     them to (−c, c) (Gemma 2's stability trick);
//   - final-logit soft-cap: the output logits are passed through the same soft-cap
//     with c = FinalLogitCap;
//   - query_pre_attn_scalar: the attention scores are scaled by 1/√QueryPreAttnScalar
//     (NOT 1/√head_dim in general; the two coincide when the scalar equals head_dim).
//
// Because OpMHA cannot insert a soft-cap between the score scaling and the mask,
// [Gemma2.Forward] builds attention from primitives (matmul/transpose/softcap/
// softmax) per head, exactly mirroring the reference mhaKernel's scale/mask/softmax
// semantics with the soft-cap spliced onto the scaled scores. Gemma 2 also alternates
// sliding-window attention every other layer; for short prompts (≤ window) the window
// never triggers, so it is intentionally omitted here (documented, not implemented).
//
// Load a Hugging Face Gemma2ForCausalLM checkpoint with [Gemma2FromHF]. As with
// [Gemma], GoAI's exact-erf OpGELU vs Gemma's tanh-approx gelu_pytorch_tanh leaves a
// small approximation residual, bounded by the parity test.
type Gemma2 struct {
	Config    Gemma2Config
	TokEmb    *tensor.Tensor // [vocab, dim] tied token embedding
	Blocks    []*Gemma2Block
	FinalNorm *nn.RMSNorm

	outT *tensor.Tensor // cached [dim, vocab] tied-head transpose (see tiedHead)
}

// tiedHead returns the cached [dim, vocab] tied-head transpose (computed once) — the
// per-call transpose cost ~220ms at Llama-scale dims (§V22). Call
// [Gemma2.RefreshTiedHead] after mutating TokEmb in place (fine-tuning).
func (m *Gemma2) tiedHead() *tensor.Tensor {
	if m.outT == nil {
		m.outT = transpose2D(m.TokEmb)
	}
	return m.outT
}

// RefreshTiedHead drops the cached tied-head transpose so the next Forward/DecodeStep
// recomputes it from the current TokEmb values.
func (m *Gemma2) RefreshTiedHead() { m.outT = nil }

// Gemma2Config fixes the model geometry. Dim, Vocab, Layers, FFN and HeadDim are
// inferred from the checkpoint by [Gemma2FromHF]; the remaining fields come from
// config.json. QueryPreAttnScalar, AttnLogitCap and FinalLogitCap are the three
// Gemma 2-specific scalars; leaving a cap ≤ 0 disables that soft-cap.
type Gemma2Config struct {
	Vocab    int
	Ctx      int
	Dim      int // d_model (embedding width)
	Heads    int // query heads
	KVHeads  int // key/value heads (GQA); 0 → Heads
	HeadDim  int // per-head width (may differ from Dim/Heads)
	Layers   int
	FFN      int     // GeGLU inner width
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE θ; 0 → 10000
	// QueryPreAttnScalar sets the pre-softmax score scale to 1/√QueryPreAttnScalar
	// (config.query_pre_attn_scalar). 0 → HeadDim (recovering the standard 1/√head_dim).
	QueryPreAttnScalar float64
	// AttnLogitCap is the attention-score soft-cap magnitude (attn_logit_softcapping);
	// ≤ 0 disables the score soft-cap.
	AttnLogitCap float64
	// FinalLogitCap is the output-logit soft-cap magnitude (final_logit_softcapping);
	// ≤ 0 disables the final soft-cap.
	FinalLogitCap float64
}

// Gemma2Block is one Gemma 2 block with the four sandwich RMSNorms. Every gain already
// carries the +1 offset (folded at load by gemmaRMS), so nn.RMSNorm applies Gemma's
// (1+w) directly. The FFN is a GeGLU (nn.GLU with the GELU gate).
type Gemma2Block struct {
	InputNorm      *nn.RMSNorm    // input_layernorm (pre-attention)
	Wq, Wk, Wv, Wo *tensor.Tensor // [dim, heads·head_dim] … [heads·head_dim, dim]
	PostAttnNorm   *nn.RMSNorm    // post_attention_layernorm (post-attention, pre-residual)
	PreFFNNorm     *nn.RMSNorm    // pre_feedforward_layernorm
	FFN            *nn.GLU        // GeGLU
	PostFFNNorm    *nn.RMSNorm    // post_feedforward_layernorm (post-FFN, pre-residual)
}

func (c Gemma2Config) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// queryScale returns the pre-softmax score multiplier 1/√QueryPreAttnScalar, falling
// back to 1/√HeadDim when the scalar is unset.
func (c Gemma2Config) queryScale() float64 {
	q := c.QueryPreAttnScalar
	if q <= 0 {
		q = float64(c.HeadDim)
	}
	return 1.0 / math.Sqrt(q)
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Gemma2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	return m.forwardCapture(ctx, tokens, nil)
}

// forwardCapture is Forward with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// full-width POST-RoPE k and raw post-projection v [seq, kvHeads·headDim] —
// exactly the rows [Gemma2.DecodeStep]'s cappedDecodeAttention appends per token
// (k rotated, v unrotated, both BEFORE the per-head slicing of the soft-capped
// attention), which is what lets [Gemma2.Prefill] seed a cache from ONE batched
// pass over the prompt. The tensors handed to capture are the same ones the
// ongoing forward consumes (callers must not mutate them; the cache copies rows
// into its own buffers). A nil capture is the plain forward: no extra ops are
// recorded, so tape/training semantics of Forward are byte-identical to before
// this hook existed.
func (m *Gemma2) forwardCapture(ctx *backend.Context, tokens []int, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Gemma2 prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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
	// Gemma scales the embeddings by √dim (the "normalizer") before the blocks; the
	// tied LM head uses the UNSCALED table, so this is a runtime scalar on the
	// residual stream only (mirrors [Gemma.Forward]).
	scale := tensor.New(tensor.F64, tensor.Shape{})
	scale.Storage().F64()[0] = math.Sqrt(float64(m.Config.Dim))
	if x, err = exec1(ctx, backend.OpMul, nil, x, scale); err != nil {
		return nil, err
	}

	for l, b := range m.Blocks {
		// Attention sublayer: input_layernorm → capped attention →
		// post_attention_layernorm → add residual.
		xb, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var kvTap func(k, v *tensor.Tensor)
		if capture != nil {
			kvTap = func(k, v *tensor.Tensor) { capture(l, k, v) }
		}
		a, err := m.cappedAttention(ctx, b, xb, seq, kvTap)
		if err != nil {
			return nil, err
		}
		if a, err = b.PostAttnNorm.Forward(ctx, a); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// FFN sublayer: pre_feedforward_layernorm → GeGLU →
		// post_feedforward_layernorm → add residual.
		xf, err := b.PreFFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if ff, err = b.PostFFNNorm.Forward(ctx, ff); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Tied LM head: logits = hidden · embedᵀ.
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.tiedHead())
	if err != nil {
		return nil, err
	}
	// Final-logit soft-cap.
	if m.Config.FinalLogitCap > 0 {
		logits, err = exec1(ctx, backend.OpSoftCap, backend.SoftCapAttrs{Cap: m.Config.FinalLogitCap}, logits)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// cappedAttention computes Gemma 2's soft-capped multi-head attention from primitives.
// It mirrors the reference mhaKernel numerics exactly — per-head scaled dot product,
// causal −∞ mask for j>i, stable softmax over keys, then ·V — but scales by
// 1/√QueryPreAttnScalar (not 1/√head_dim) and inserts the attention-logit soft-cap
// c·tanh(scores/c) on the SCALED scores, BEFORE the mask (matching HF Gemma2Attention:
// scores·scale → softcap → +mask → softmax). GQA is handled by mapping query head h to
// KV head h/(heads/kv_heads). Sliding-window attention is intentionally not applied
// (no-op for prompts within the window).
//
// capture, when non-nil, receives the full-width POST-RoPE k and the raw
// (unrotated) v [seq, kvHeads·headDim] right before the per-head slicing —
// exactly what cappedDecodeAttention appends to the KV-cache per token — so
// Prefill can seed a cache during a batched pass. nil is the plain attention.
func (m *Gemma2) cappedAttention(ctx *backend.Context, b *Gemma2Block, xb *tensor.Tensor, seq int, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	hd := cfg.HeadDim
	rep := cfg.Heads / kv

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
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, k); err != nil {
		return nil, err
	}
	if capture != nil {
		capture(k, v)
	}

	// Pre-softmax score scale as a rank-0 scalar (broadcast over [seq,seq]).
	scaleT := tensor.New(tensor.F64, tensor.Shape{})
	scaleT.Storage().F64()[0] = cfg.queryScale()

	// Causal additive mask: 0 for j≤i, −∞ for j>i.
	mask := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	ms := mask.Storage().F64()
	for i := range seq {
		for j := range seq {
			if j > i {
				ms[i*seq+j] = math.Inf(-1)
			}
		}
	}

	heads := make([]*tensor.Tensor, cfg.Heads)
	for h := range cfg.Heads {
		kvHead := h / rep
		qh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * hd, End: (h + 1) * hd}, q)
		if err != nil {
			return nil, err
		}
		kh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: kvHead * hd, End: (kvHead + 1) * hd}, k)
		if err != nil {
			return nil, err
		}
		vh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: kvHead * hd, End: (kvHead + 1) * hd}, v)
		if err != nil {
			return nil, err
		}
		// scores = Qh·Khᵀ  [seq,seq]
		khT, err := exec1(ctx, backend.OpTranspose, nil, kh)
		if err != nil {
			return nil, err
		}
		scores, err := exec1(ctx, backend.OpMatMul, nil, qh, khT)
		if err != nil {
			return nil, err
		}
		// scores·scale (query_pre_attn_scalar^-0.5)
		if scores, err = exec1(ctx, backend.OpMul, nil, scores, scaleT); err != nil {
			return nil, err
		}
		// attention-logit soft-cap on the scaled scores, before the mask
		if cfg.AttnLogitCap > 0 {
			if scores, err = exec1(ctx, backend.OpSoftCap, backend.SoftCapAttrs{Cap: cfg.AttnLogitCap}, scores); err != nil {
				return nil, err
			}
		}
		// + causal mask, then softmax over keys (last axis)
		if scores, err = exec1(ctx, backend.OpAdd, nil, scores, mask); err != nil {
			return nil, err
		}
		probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
		if err != nil {
			return nil, err
		}
		oh, err := exec1(ctx, backend.OpMatMul, nil, probs, vh) // [seq,hd]
		if err != nil {
			return nil, err
		}
		heads[h] = oh
	}
	// Concatenate head outputs → [seq, heads·head_dim], then o_proj.
	concat, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, concat, b.Wo)
}

// Params returns every trainable tensor for optimizers.
func (m *Gemma2) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb, m.FinalNorm.Gamma}
	for _, b := range m.Blocks {
		ps = append(ps, b.InputNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo)
		ps = append(ps, b.PostAttnNorm.Gamma, b.PreFFNNorm.Gamma, b.PostFFNNorm.Gamma)
		ps = append(ps, b.FFN.Params()...)
	}
	return ps
}
