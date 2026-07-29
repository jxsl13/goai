package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantMPT is an [MPT] whose projection weights stay QUANTIZED (nn.QuantLinear) and are
// never materialized as full-precision matrices — the memory-efficient form of a
// llama.cpp-quantized MPT checkpoint, the mpt sibling of [QuantLlama] / [QuantGemma].
// Its forward mirrors MPT.Forward faithfully, all four MPT departures included:
//
//   - ALiBi position bias, NO RoPE and NO positional embedding: attention runs with
//     [backend.AttnAttrs].ALiBi set, so [backend.OpMHA] adds the per-head
//     linear-distance penalty slopeₕ·(j−i) (alibi_bias_max=8 slope family,
//     [backend.ALiBiSlopes]) to the pre-softmax scores. The quantized q/k/v are used
//     UNROTATED — position enters solely through the bias, exactly like the float path.
//   - WEIGHT-ONLY LayerNorm (no_bias=True): norm_1/norm_2/norm_f carry only γ; the
//     mean-centered LayerNorm runs with β = 0. The quantized twin keeps them as f32
//     γ + zero-β [nn.LayerNorm]s so they compose with the f32 activation stream.
//   - BIAS-FREE projections: every QuantLinear runs bare (no bias add anywhere) —
//     Wqkv, out_proj, ffn.up_proj and ffn.down_proj are all bias-free in the
//     no_bias=True form this type models.
//   - TIED LM head: like [QuantGemma], Out wraps the SAME quantized token-embedding
//     bytes as a QuantLinear (logits = hidden · dequant(table)ᵀ), while TokEmb holds
//     the dequantized f32 table for the embedding lookup — one tensor, two views.
//
// Dtype discipline (the §B quant-bias lesson): the quantized residual stream is f32 —
// QuantLinear outputs f32 activations — so every small tensor that touches it is f32
// too: the LayerNorm γ (and its zero β) and the dequantized embedding table. Real
// llama.cpp-quantized mpt files store the 1-D norms as F32 (1-D tensors are never
// block-quantized), so the GGUF path and [QuantizeMPT] land on identical bytes.
// Inference-only: the quantized weights are frozen and bypass the tape.
//
// Load a llama.cpp-quantized checkpoint with [QuantMPTFromGGUF], or quantize a float
// model with [QuantizeMPT].
type QuantMPT struct {
	Config    MPTConfig        // model geometry, shared with float MPT (see MPTConfig)
	TokEmb    *tensor.Tensor   // [vocab, dim] f32 DEQUANTIZED embedding table (lookup only)
	Blocks    []*QuantMPTBlock // the quantized MPT blocks
	FinalNorm *nn.LayerNorm    // norm_f: weight-only (f32 γ, zero β)
	Out       *nn.QuantLinear  // tied LM head: the quantized token_embd bytes (or an untied output.weight)
}

// QuantMPTBlock is one sequential-residual QuantMPT block: f32 weight-only LayerNorms
// (γ, zero β — the no_bias=True form) gating quantized BIAS-FREE attention projections
// (the fused Wqkv unpacked into per-projection QuantLinears) and a quantized bias-free
// 2-layer GELU MLP. There are no bias vectors anywhere — canonical MPT checkpoints
// carry none, and GoAI's [MPT] models exactly that form.
type QuantMPTBlock struct {
	Norm1, Norm2   *nn.LayerNorm   // norm_1 (pre-attention) / norm_2 (pre-MLP), weight-only f32
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (Wqkv thirds + out_proj), no bias
	Wup, Wdown     *nn.QuantLinear // quantized ffn.up_proj / ffn.down_proj, no bias
}

// QuantizeMPT builds a QuantMPT from a float MPT by quantizing every 2-D projection
// (the q/k/v thirds, out_proj, ffn up/down) to qt — the projections carry the bulk of
// the weights and compute, so this is where quantization pays off. The tied head is
// where MPT differs from the untied [QuantizeStarCoder2]: the token embedding is
// quantized ONCE and the SAME bytes serve as the head (Out), while TokEmb becomes the
// table those bytes dequantize to — quantizing the table but embedding from the exact
// float one would silently desynchronize the two tied views; using the dequantized one
// reproduces exactly what [QuantMPTFromGGUF] loads (the exact-anchor gate). Everything
// 1-D stays f32: the weight-only LayerNorm γ (with its zero β), matching llama.cpp's
// on-disk convention of never block-quantizing 1-D tensors. Each projection's inner
// dimension must be a multiple of qt's block size (32 for Q8_0/Q4_0, 256 for the
// k-quants) — for the tied head that means Dim itself.
func QuantizeMPT(m *MPT, qt gguf.QuantType) (*QuantMPT, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	vocab, dim := m.TokEmb.Shape()[0], m.TokEmb.Shape()[1]
	headBytes, err := gguf.Quantize(m.TokEmb, qt) // [vocab, dim] IS the head's [out, in]
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeMPT token embedding: %w", err)
	}
	emb, err := (gguf.QuantTensor{Data: headBytes, GGType: uint32(qt), Shape: tensor.Shape{vocab, dim}}).Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeMPT token embedding: %w", err)
	}
	q := &QuantMPT{
		Config:    m.Config,
		TokEmb:    emb, // the dequantized view of the SAME bytes the tied head consumes
		FinalNorm: f32LayerNorm(m.FinalNorm),
		Out:       &nn.QuantLinear{Weight: headBytes, QT: qt, In: dim, Out: vocab},
	}
	for _, b := range m.Blocks {
		qb := &QuantMPTBlock{
			Norm1: f32LayerNorm(b.Norm1), // weight-only: the float β is the zero vector, cloned as f32 zeros
			Norm2: f32LayerNorm(b.Norm2),
		}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMPT Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMPT Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMPT Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMPT out_proj: %w", err)
		}
		if qb.Wup, err = mkQ(b.Wup); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMPT up_proj: %w", err)
		}
		if qb.Wdown, err = mkQ(b.Wdown); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMPT down_proj: %w", err)
		}
		q.Blocks = append(q.Blocks, qb)
	}
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It
// mirrors MPT.Forward exactly — embedding lookup (no positional embedding, no scale),
// then per block the sequential residual
//
//	x = x + attn(norm_1(x)); x = x + ffn(norm_2(x))
//
// with bias-free quantized projections, NO RoPE (position enters through OpMHA's
// per-head ALiBi bias) and standard causal MHA (no GQA), then norm_f and the quantized
// tied head — but every projection is a quantized in-kernel matmul, all activations f32.
func (m *QuantMPT) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: MPT prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
	// Causal MHA with the ALiBi distance bias — the float forward's exact attrs.
	attn := backend.AttnAttrs{Heads: cfg.Heads, Causal: true, ALiBi: true}
	for _, b := range m.Blocks {
		// Attention sublayer: norm_1 → bias-free q/k/v (NO rotation) → causal
		// ALiBi MHA → bias-free out_proj → residual add.
		an, err := b.Norm1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := b.Wq.Forward(ctx, an)
		if err != nil {
			return nil, err
		}
		k, err := b.Wk.Forward(ctx, an)
		if err != nil {
			return nil, err
		}
		v, err := b.Wv.Forward(ctx, an)
		if err != nil {
			return nil, err
		}
		a, err := exec3(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		if a, err = b.Wo.Forward(ctx, a); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// MLP sublayer: norm_2 → bias-free GELU 2-layer → residual add.
		fn, err := b.Norm2.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, fn)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// mlp runs the bias-free 2-layer GELU feed-forward on the normalized input fn
// [seq, dim]: up_proj → exact-erf GELU → down_proj, the quantized twin of MPT.mlp.
func (m *QuantMPT) mlp(ctx *backend.Context, b *QuantMPTBlock, fn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := b.Wup.Forward(ctx, fn)
	if err != nil {
		return nil, err
	}
	if h, err = exec1a(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return b.Wdown.Forward(ctx, h)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this QuantMPT.
// Reuses [MPTCache] — the cache structure is identical to the float model's (raw,
// unrotated k/v rows; MPT has no RoPE); only the projections differ.
func (m *QuantMPT) NewCache() *MPTCache {
	return &MPTCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token (MPT has no
// embedding scale and no positional embedding — position enters through ALiBi).
func (m *QuantMPT) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized MPT by one token at absolute position pos,
// appending its raw post-projection K/V to the cache and returning the next-token
// logits [1, vocab]. It mirrors MPT.DecodeStep exactly — single-token embedding,
// bias-free quantized projections, NO RoPE (the cached rows are position-independent;
// [backend.OpMHA]'s ALiBi bias slopeₕ·(j−pos) is recovered automatically from the
// query/key length gap off = sk−sq = pos), the single query attending to the whole
// cache without a causal mask, the bias-free GELU MLP, norm_f, tied head — so a
// KV-cache decode matches the full Forward (same kernel sequence per row).
// Inference-only, like the rest of the type.
func (m *QuantMPT) DecodeStep(ctx *backend.Context, cache *MPTCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// Single query vs all cached keys: no causal mask (every cached key is at pos ≤
	// this token); ALiBi stays on, correctly biased at absolute pos via the sq/sk gap.
	attn := backend.AttnAttrs{Heads: cfg.Heads, Causal: false, ALiBi: true}
	for l, b := range m.Blocks {
		an, err := b.Norm1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := b.Wq.Forward(ctx, an)
		if err != nil {
			return nil, err
		}
		k, err := b.Wk.Forward(ctx, an)
		if err != nil {
			return nil, err
		}
		v, err := b.Wv.Forward(ctx, an)
		if err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		a, err := exec3(ctx, backend.OpMHA, attn, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = b.Wo.Forward(ctx, a); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		fn, err := b.Norm2.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, fn)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on the
// quantized model, using the KV-cache (each step is one token, not a full re-forward),
// and returns prompt+new. The sampler s selects each token (Greedy() for deterministic
// argmax). Stops at the context limit. The same shape as [QuantLlama.Generate].
func (m *QuantMPT) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
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
		if gc.stopEOS(next, s) {
			break
		}
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
// projections (attention, MLP, the tied head). Idempotent; call it when done with the
// model to release GPU memory promptly.
func (m *QuantMPT) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, b := range m.Blocks {
		for _, l := range []*nn.QuantLinear{b.Wq, b.Wk, b.Wv, b.Wo, b.Wup, b.Wdown} {
			if l != nil {
				note(l.Close())
			}
		}
	}
	if m.Out != nil {
		note(m.Out.Close())
	}
	return first
}
