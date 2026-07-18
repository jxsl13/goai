package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantGemma2 is a [Gemma2] whose projection weights stay QUANTIZED (nn.QuantLinear)
// and are never materialized as full-precision matrices — the memory-efficient form of
// a llama.cpp-quantized Gemma 2 checkpoint, the gemma2 sibling of [QuantGemma]. Its
// forward mirrors Gemma2.Forward faithfully, all four Gemma 2 departures included:
//
//   - SANDWICH norms: each block runs FOUR (1+γ)-carrying f32 RMSNorms —
//     input_layernorm → capped attention → post_attention_layernorm → +residual, then
//     pre_feedforward_layernorm → GeGLU → post_feedforward_layernorm → +residual;
//   - attention-logit soft-cap: the scaled pre-softmax scores pass through
//     y=c·tanh(x/c) (c = AttnLogitCap) BEFORE the causal mask. OpMHA cannot splice a
//     soft-cap between scale and mask, so the attention is built from the same
//     primitive sequence as the float [Gemma2] — per-head matmul/scale/OpSoftCap/
//     mask/softmax — on the f32 activation stream (OpSoftCap is activation-level and
//     dtype-agnostic; the quant path reuses the identical op);
//   - final-logit soft-cap: the output logits pass through the same soft-cap with
//     c = FinalLogitCap;
//   - query_pre_attn_scalar: scores are scaled by 1/√QueryPreAttnScalar (NOT
//     1/√head_dim in general — llama.cpp's 27B derives Dim/Heads instead).
//
// Every Gemma (v1) trait carries over too: the √dim embedding "normalizer" applied at
// runtime to the residual stream only, RoPE, grouped-query attention, the GeGLU
// ([QuantGeGLU]) FFN, decoupled head_dim, and the LM head TIED to the token embedding:
// Out wraps the SAME quantized token-embedding Q-block bytes as an [nn.QuantLinear]
// (logits = hidden · dequant(table)ᵀ), while TokEmb holds the dequantized f32 table
// those bytes decode to, used only for the embedding lookup — one on-disk tensor, two
// views, exactly as in a real quantized gemma2 GGUF. Activations are f32 —
// quantized-inference precision, which also engages the f32-only GPU kernels.
// Inference-only: the quantized weights are frozen and bypass the tape.
//
// Sliding-window semantics follow the float type: [QuantGemma2FromGGUF] clamps Ctx to
// min(context_length, sliding_window), so full attention is bit-identical to
// llama.cpp's alternating-window graph for every prompt the model accepts, and longer
// prompts are rejected rather than silently mis-attended.
//
// Load a llama.cpp-quantized checkpoint with [QuantGemma2FromGGUF], or quantize a
// float model with [QuantizeGemma2].
type QuantGemma2 struct {
	Config    Gemma2Config
	TokEmb    *tensor.Tensor // [vocab, dim] f32 DEQUANTIZED embedding table (lookup only)
	Blocks    []*QuantGemma2Block
	FinalNorm *nn.RMSNorm     // final pre-logits RMSNorm (f32 gain, +1 pre-folded)
	Out       *nn.QuantLinear // tied LM head: the quantized token_embd bytes, In=dim Out=vocab
}

// QuantGemma2Block is one sandwich-normed QuantGemma2 block: four f32 RMSNorm gains
// (all already carrying Gemma's +1 offset, the same in-memory convention as
// [Gemma2Block]) around quantized attention projections and a quantized GeGLU FFN. The
// projections keep Gemma 2's decoupled head width: Wq maps dim→heads·head_dim and Wo
// back, with head_dim free to differ from dim/heads.
type QuantGemma2Block struct {
	InputNorm      *nn.RMSNorm     // input_layernorm (pre-attention, f32 gain)
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (bias-free)
	PostAttnNorm   *nn.RMSNorm     // post_attention_layernorm (post-attention, pre-residual)
	PreFFNNorm     *nn.RMSNorm     // pre_feedforward_layernorm
	FFN            *QuantGeGLU     // quantized GeGLU feed-forward
	PostFFNNorm    *nn.RMSNorm     // post_feedforward_layernorm (post-FFN, pre-residual)
}

// QuantizeGemma2 builds a QuantGemma2 from a float Gemma2 by quantizing every linear
// projection to qt (the ggml block layout) — the projections carry the bulk of the
// weights and compute, so this is where quantization pays off. All FOUR sandwich-norm
// gains per block plus the final norm stay in f32 (tiny and precision-sensitive; they
// keep carrying Gemma's pre-folded +1 unchanged — real llama.cpp files never
// block-quantize 1-D tensors). Each projection's inner dimension must be a multiple of
// qt's block size (32 for Q8_0/Q4_0, 256 for the k-quants) — for the tied head that
// means Dim itself.
//
// Like [QuantizeGemma] (and unlike [QuantizeLlama]), the token embedding IS quantized:
// Gemma 2's LM head is the embedding table, so the [vocab, dim] table — already the
// head's [out, in] layout — is encoded once and the SAME bytes serve as the head
// (Out), while TokEmb becomes the table those bytes dequantize to. Keeping the
// original float table for lookups would desynchronize the two tied views; using the
// dequantized one reproduces exactly what [QuantGemma2FromGGUF] yields from a file
// quantized from the same weights.
func QuantizeGemma2(m *Gemma2, qt gguf.QuantType) (*QuantGemma2, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	// Tied head: the [vocab, dim] table is already [out, in] — no transpose.
	vocab, dim := m.TokEmb.Shape()[0], m.TokEmb.Shape()[1]
	headBytes, err := gguf.Quantize(m.TokEmb, qt)
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeGemma2 token embedding: %w", err)
	}
	emb, err := (gguf.QuantTensor{Data: headBytes, GGType: uint32(qt), Shape: tensor.Shape{vocab, dim}}).Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeGemma2 token embedding: %w", err)
	}
	q := &QuantGemma2{
		Config:    m.Config,
		TokEmb:    emb,
		FinalNorm: f32RMSNorm(m.FinalNorm),
		Out:       &nn.QuantLinear{Weight: headBytes, QT: qt, In: dim, Out: vocab},
	}
	for _, b := range m.Blocks {
		qb := &QuantGemma2Block{
			InputNorm:    f32RMSNorm(b.InputNorm),
			PostAttnNorm: f32RMSNorm(b.PostAttnNorm),
			PreFFNNorm:   f32RMSNorm(b.PreFFNNorm),
			PostFFNNorm:  f32RMSNorm(b.PostFFNNorm),
		}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma2 Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma2 Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma2 Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma2 Wo: %w", err)
		}
		gate, err := mkQ(b.FFN.Wgate)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma2 ffn_gate: %w", err)
		}
		up, err := mkQ(b.FFN.Wup)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma2 ffn_up: %w", err)
		}
		down, err := mkQ(b.FFN.Wdown)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma2 ffn_down: %w", err)
		}
		qb.FFN = &QuantGeGLU{Gate: gate, Up: up, Down: down}
		q.Blocks = append(q.Blocks, qb)
	}
	return q, nil
}

// embedScale returns the √dim "normalizer" as an f32 scalar tensor — the same runtime
// broadcast Gemma2.Forward applies right after the embedding lookup, computed at the
// activations' f32 precision (llama.cpp likewise scales with sqrtf). It multiplies the
// residual stream only; the tied head keeps using the unscaled table.
func (m *QuantGemma2) embedScale() *tensor.Tensor {
	s := tensor.New(tensor.F32, tensor.Shape{})
	s.Storage().F32()[0] = float32(math.Sqrt(float64(m.Config.Dim)))
	return s
}

// queryScaleF32 returns 1/√QueryPreAttnScalar as a rank-0 f32 tensor — the quant
// stream's pre-softmax score multiplier (the float path holds the same value in f64).
func (m *QuantGemma2) queryScaleF32() *tensor.Tensor {
	s := tensor.New(tensor.F32, tensor.Shape{})
	s.Storage().F32()[0] = float32(m.Config.queryScale())
	return s
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It
// mirrors Gemma2.Forward exactly — embedding lookup, ×√dim on the residual stream, the
// four-norm sandwich blocks with (1+γ)-carrying f32 gains, the soft-capped primitive
// attention at the 1/√QueryPreAttnScalar scale, GeGLU, final norm, tied head, and the
// final-logit soft-cap — but every projection is a quantized in-kernel matmul, all
// activations f32.
func (m *QuantGemma2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: Gemma2 prompt length %d outside (0,%d]", seq, cfg.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= cfg.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, cfg.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
	if err != nil {
		return nil, err
	}
	// √dim normalizer on the residual stream only (the tied head stays unscaled) — the
	// same lookup-then-scale order as Gemma2.Forward.
	if x, err = exec1(ctx, backend.OpMul, nil, x, m.embedScale()); err != nil {
		return nil, err
	}
	for _, b := range m.Blocks {
		// Attention sublayer: input_layernorm → capped attention →
		// post_attention_layernorm → add residual.
		xb, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.cappedAttention(ctx, b, xb, seq)
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
	// Tied LM head: logits = hidden · dequant(table)ᵀ, straight from the Q-blocks.
	logits, err := m.Out.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	// Final-logit soft-cap, the same OpSoftCap primitive as the float path.
	if cfg.FinalLogitCap > 0 {
		logits, err = exec1(ctx, backend.OpSoftCap, backend.SoftCapAttrs{Cap: cfg.FinalLogitCap}, logits)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// cappedAttention computes Gemma 2's soft-capped multi-head attention on the f32 quant
// stream, replicating [Gemma2.cappedAttention]'s primitive sequence exactly — per-head
// Qh·Khᵀ, ×1/√QueryPreAttnScalar, OpSoftCap on the SCALED scores BEFORE the causal
// mask (matching HF Gemma2Attention: scores·scale → softcap → +mask → softmax), stable
// softmax over keys, ·V, concat, o_proj — with the q/k/v/o matmuls swapped for
// quantized in-kernel projections and the mask/scale constants built at f32. GQA maps
// query head h to KV head h/(heads/kv_heads). Sliding-window attention is intentionally
// not applied, exactly as in the float type (a no-op for prompts within the window,
// which the Ctx clamp guarantees for GGUF-loaded models).
func (m *QuantGemma2) cappedAttention(ctx *backend.Context, b *QuantGemma2Block, xb *tensor.Tensor, seq int) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	hd := cfg.HeadDim
	rep := cfg.Heads / kv

	q, err := b.Wq.Forward(ctx, xb)
	if err != nil {
		return nil, err
	}
	k, err := b.Wk.Forward(ctx, xb)
	if err != nil {
		return nil, err
	}
	v, err := b.Wv.Forward(ctx, xb)
	if err != nil {
		return nil, err
	}
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, k); err != nil {
		return nil, err
	}

	scaleT := m.queryScaleF32()

	// Causal additive mask: 0 for j≤i, −∞ for j>i (f32, the activations' dtype).
	mask := tensor.New(tensor.F32, tensor.Shape{seq, seq})
	ms := mask.Storage().F32()
	negInf := float32(math.Inf(-1))
	for i := range seq {
		for j := range seq {
			if j > i {
				ms[i*seq+j] = negInf
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
	// Concatenate head outputs → [seq, heads·head_dim], then the quantized o_proj.
	concat, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	if err != nil {
		return nil, err
	}
	return b.Wo.Forward(ctx, concat)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this
// QuantGemma2. Reuses [Gemma2Cache] — the cache structure is identical to the float
// model's (post-RoPE keys, raw values); only the projections differ.
func (m *QuantGemma2) NewCache() *Gemma2Cache {
	return &Gemma2Cache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the √dim-scaled f32 embedding [1, dim] of a single token — the
// quantized twin of Gemma2.embedOne: lookup from the dequantized table, then the same
// runtime ×√dim on the residual stream (the tied head keeps the unscaled bytes).
func (m *QuantGemma2) embedOne(ctx *backend.Context, token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	x := embedRow(m.TokEmb, token, m.Config.Dim)
	return exec1(ctx, backend.OpMul, nil, x, m.embedScale())
}

// DecodeStep advances the quantized Gemma 2 by one token at absolute position pos,
// appending its post-RoPE K and raw V to the cache and returning the next-token
// logits [1, vocab]. It mirrors Gemma2.DecodeStep exactly — √dim-scaled single-token
// embedding, the sandwich-normed blocks with the single query attending to the whole
// cache through the soft-capped attention (no causal mask needed), GeGLU, final norm,
// tied head, final-logit soft-cap — so a KV-cache decode matches the full Forward
// (same primitive sequence per row). Inference-only, like the rest of the type.
func (m *QuantGemma2) DecodeStep(ctx *backend.Context, cache *Gemma2Cache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	x, err := m.embedOne(ctx, token)
	if err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		// Attention sublayer: input_layernorm → capped attention →
		// post_attention_layernorm → add residual.
		xb, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.cappedDecodeAttention(ctx, b, xb, cache, l, pos)
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
	// Tied LM head from the Q-block bytes, then the final-logit soft-cap.
	logits, err := m.Out.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	if cfg.FinalLogitCap > 0 {
		logits, err = exec1(ctx, backend.OpSoftCap, backend.SoftCapAttrs{Cap: cfg.FinalLogitCap}, logits)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// cappedDecodeAttention computes the soft-capped attention for the single new query
// token against the cached keys/values, appending this token's rotated k and raw v to
// the cache first — the quantized twin of [Gemma2.cappedDecodeAttention], same
// primitive numerics per head (scores·scale → OpSoftCap → softmax → ·V) but for ONE
// query row and WITHOUT the causal mask: a single query legitimately attends to all
// cached keys, so the mask is a no-op for it. GQA maps query head h to KV head
// h/(heads/kv).
func (m *QuantGemma2) cappedDecodeAttention(ctx *backend.Context, b *QuantGemma2Block, xb *tensor.Tensor, cache *Gemma2Cache, l, pos int) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	hd := cfg.HeadDim
	rep := cfg.Heads / kv

	q, err := b.Wq.Forward(ctx, xb)
	if err != nil {
		return nil, err
	}
	k, err := b.Wk.Forward(ctx, xb)
	if err != nil {
		return nil, err
	}
	v, err := b.Wv.Forward(ctx, xb)
	if err != nil {
		return nil, err
	}
	// RoPE the single token at its absolute position, then append k,v to the cache.
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}, k); err != nil {
		return nil, err
	}
	kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
	cache.K[l], cache.V[l] = kNew, vNew

	scaleT := m.queryScaleF32()

	heads := make([]*tensor.Tensor, cfg.Heads)
	for h := range cfg.Heads {
		kvHead := h / rep
		qh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * hd, End: (h + 1) * hd}, q)
		if err != nil {
			return nil, err
		}
		kh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: kvHead * hd, End: (kvHead + 1) * hd}, kNew)
		if err != nil {
			return nil, err
		}
		vh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: kvHead * hd, End: (kvHead + 1) * hd}, vNew)
		if err != nil {
			return nil, err
		}
		// scores = qh·khᵀ  [1,sk]
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
		// attention-logit soft-cap on the scaled scores (no mask for a single query)
		if cfg.AttnLogitCap > 0 {
			if scores, err = exec1(ctx, backend.OpSoftCap, backend.SoftCapAttrs{Cap: cfg.AttnLogitCap}, scores); err != nil {
				return nil, err
			}
		}
		probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
		if err != nil {
			return nil, err
		}
		oh, err := exec1(ctx, backend.OpMatMul, nil, probs, vh) // [1,hd]
		if err != nil {
			return nil, err
		}
		heads[h] = oh
	}
	// Concatenate head outputs → [1, heads·head_dim], then the quantized o_proj.
	concat, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	if err != nil {
		return nil, err
	}
	return b.Wo.Forward(ctx, concat)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on the
// quantized model, using the KV-cache (each step is one token, not a full re-forward),
// and returns prompt+new. The sampler s selects each token (Greedy() for deterministic
// argmax). Stops at the context limit. The same shape as [QuantGemma.Generate].
func (m *QuantGemma2) Generate(prompt []int, maxNew int, s TokenSampler) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}
	ctx := backend.NewContext()
	cache := m.NewCache()
	out := append([]int(nil), prompt...)
	var logits *tensor.Tensor
	pos := 0
	for _, tok := range prompt {
		l, err := m.DecodeStep(ctx, cache, tok, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	for range maxNew {
		if pos >= m.Config.Ctx {
			break
		}
		next := s.SampleWithHistory(rowLogits(logits), out)
		out = append(out, next)
		l, err := m.DecodeStep(ctx, cache, next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}

// Close frees every device-resident weight buffer held by the model's quantized
// projections (attention, FFN, the tied head). Idempotent; call it when done with the
// model to release GPU memory promptly.
func (m *QuantGemma2) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, b := range m.Blocks {
		for _, l := range []*nn.QuantLinear{b.Wq, b.Wk, b.Wv, b.Wo} {
			if l != nil {
				note(l.Close())
			}
		}
		if b.FFN != nil {
			note(b.FFN.Close())
		}
	}
	if m.Out != nil {
		note(m.Out.Close())
	}
	return first
}
