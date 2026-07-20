package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantGemma is a [Gemma] (v1) whose projection weights stay QUANTIZED (nn.QuantLinear)
// and are never materialized as full-precision matrices — the memory-efficient form of a
// llama.cpp-quantized Gemma checkpoint, the gemma sibling of [QuantLlama] (§T149). Its
// forward mirrors Gemma.Forward faithfully: the √dim embedding "normalizer" on the
// residual stream, (1+γ) RMSNorm via pre-folded f32 gains, RoPE, grouped-query
// attention, the GeGLU (GELU-gated) FFN, and the LM head tied to the token embedding —
// but every matmul is an in-kernel dequantized [nn.QuantLinear] that runs on the active
// accelerator when it supports the quant type (else the CPU fallback). Activations are
// f32 — quantized-inference precision, which is also what engages the f32-only GPU
// kernels. Inference-only: the quantized weights are frozen and bypass the tape.
//
// The tied head is where Gemma differs structurally from the Llama family: there is no
// separate output projection. Out wraps the SAME quantized token-embedding bytes as a
// QuantLinear (logits = hidden · dequant(table)ᵀ), while TokEmb holds the dequantized
// f32 table those bytes decode to, used only for the embedding lookup. Both views come
// from one on-disk tensor, exactly as in a real quantized gemma GGUF. The √dim scale is
// applied at runtime to the looked-up embedding only — never to the head — preserving
// Gemma's scaled-stream / unscaled-head split.
//
// Load a llama.cpp-quantized checkpoint with [QuantGemmaFromGGUF], or quantize a float
// model with [QuantizeGemma].
type QuantGemma struct {
	Config    GemmaConfig        // model geometry, shared with float Gemma (see GemmaConfig)
	TokEmb    *tensor.Tensor     // [vocab, dim] f32 DEQUANTIZED embedding table (lookup only)
	Blocks    []*QuantGemmaBlock // the quantized pre-norm Gemma blocks
	FinalNorm *nn.RMSNorm        // final pre-logits RMSNorm (f32 gain, +1 pre-folded)
	Out       *nn.QuantLinear    // tied LM head: the quantized token_embd bytes, In=dim Out=vocab
}

// QuantGemmaBlock is one pre-norm QuantGemma block: f32 RMSNorm gains (already carrying
// Gemma's +1 offset, the same in-memory convention as [GemmaBlock]) around quantized
// attention projections and a quantized GeGLU FFN. The projections keep Gemma's
// decoupled head width: Wq maps dim→heads·head_dim and Wo back, with head_dim free to
// differ from dim/heads.
type QuantGemmaBlock struct {
	AttnNorm       *nn.RMSNorm     // RMSNorm before attention (f32 gain)
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (bias-free)
	FFNNorm        *nn.RMSNorm     // RMSNorm before the FFN (f32 gain)
	FFN            *QuantGeGLU     // quantized GeGLU feed-forward
}

// QuantGeGLU is a GeGLU feed-forward network whose three projections are kept quantized
// (nn.QuantLinear) and dequantized in-kernel — the quantized twin of Gemma's GELU-gated
// [nn.GLU] variant, exactly as [nn.QuantSwiGLU] is SwiGLU's. Forward is the same
// computation, GELU(x·Gate) ⊙ (x·Up) then ·Down, but each matmul runs on the active
// accelerator (or the CPU fallback) without materializing an f32 weight matrix. Like
// the float pipeline it uses the exact erf GELU (OpGELU), so a checkpoint trained with
// the tanh approximation carries the same small, documented residual as the float path.
type QuantGeGLU struct {
	Gate *nn.QuantLinear // In=dim, Out=ffn
	Up   *nn.QuantLinear // In=dim, Out=ffn
	Down *nn.QuantLinear // In=ffn, Out=dim
}

// Forward computes GeGLU(x): GELU(x·gate) ⊙ (x·up), projected down. x is [seq, dim]
// (f32 for the accelerator path).
func (g *QuantGeGLU) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	gate, err := g.Gate.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	if gate, err = exec1(ctx, backend.OpGELU, nil, gate); err != nil {
		return nil, err
	}
	up, err := g.Up.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	h, err := exec1(ctx, backend.OpMul, nil, gate, up)
	if err != nil {
		return nil, err
	}
	return g.Down.Forward(ctx, h)
}

// Close frees any device-resident weight buffers held by the three projections. Idempotent.
func (g *QuantGeGLU) Close() error {
	var first error
	for _, l := range []*nn.QuantLinear{g.Gate, g.Up, g.Down} {
		if l != nil {
			if err := l.Close(); err != nil && first == nil {
				first = err
			}
		}
	}
	return first
}

// QuantizeGemma builds a QuantGemma from a float Gemma by quantizing every linear
// projection to qt (the ggml block layout) — the projections carry the bulk of the
// weights and compute, so this is where quantization pays off. RMSNorm gains stay in
// f32 (tiny and precision-sensitive; they keep carrying Gemma's pre-folded +1
// unchanged). Each projection's inner dimension must be a multiple of qt's block size
// (32 for Q8_0/Q4_0, 256 for the k-quants) — for the tied head that means Dim itself.
//
// Unlike [QuantizeLlama], the token embedding IS quantized: Gemma's LM head is the
// embedding table, so the [vocab, dim] table — already the head's [out, in] layout —
// is encoded once and the SAME bytes serve as the head (Out), while TokEmb becomes the
// table those bytes dequantize to. Keeping the original float table for lookups would
// desynchronize the two tied views; using the dequantized one reproduces exactly what
// [QuantGemmaFromGGUF] yields from a file quantized from the same weights.
func QuantizeGemma(m *Gemma, qt gguf.QuantType) (*QuantGemma, error) {
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
		return nil, fmt.Errorf("nlp: QuantizeGemma token embedding: %w", err)
	}
	emb, err := (gguf.QuantTensor{Data: headBytes, GGType: uint32(qt), Shape: tensor.Shape{vocab, dim}}).Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeGemma token embedding: %w", err)
	}
	q := &QuantGemma{
		Config:    m.Config,
		TokEmb:    emb,
		FinalNorm: f32RMSNorm(m.FinalNorm),
		Out:       &nn.QuantLinear{Weight: headBytes, QT: qt, In: dim, Out: vocab},
	}
	for _, b := range m.Blocks {
		qb := &QuantGemmaBlock{AttnNorm: f32RMSNorm(b.AttnNorm), FFNNorm: f32RMSNorm(b.FFNNorm)}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma Wo: %w", err)
		}
		gate, err := mkQ(b.FFN.Wgate)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma ffn_gate: %w", err)
		}
		up, err := mkQ(b.FFN.Wup)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma ffn_up: %w", err)
		}
		down, err := mkQ(b.FFN.Wdown)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGemma ffn_down: %w", err)
		}
		qb.FFN = &QuantGeGLU{Gate: gate, Up: up, Down: down}
		q.Blocks = append(q.Blocks, qb)
	}
	return q, nil
}

// embedScale returns the √dim "normalizer" as an f32 scalar tensor — the same runtime
// broadcast Gemma.Forward applies right after the embedding lookup, computed at the
// activations' f32 precision (llama.cpp likewise scales with sqrtf). It multiplies the
// residual stream only; the tied head keeps using the unscaled table.
func (m *QuantGemma) embedScale() *tensor.Tensor {
	s := tensor.New(tensor.F32, tensor.Shape{})
	s.Storage().F32()[0] = float32(math.Sqrt(float64(m.Config.Dim)))
	return s
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It
// mirrors Gemma.Forward exactly — embedding lookup, ×√dim on the residual stream,
// pre-norm blocks with (1+γ)-carrying gains, RoPE, GQA, GeGLU, final norm, tied head —
// but every projection is a quantized in-kernel matmul, all in f32.
func (m *QuantGemma) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: Gemma prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
	// same lookup-then-scale order as Gemma.Forward.
	if x, err = exec1(ctx, backend.OpMul, nil, x, m.embedScale()); err != nil {
		return nil, err
	}
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for _, b := range m.Blocks {
		// attention sublayer (bias-free)
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
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
		a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		o, err := b.Wo.Forward(ctx, a)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, o); err != nil {
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
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Tied LM head: logits = hidden · dequant(table)ᵀ, straight from the Q-blocks.
	return m.Out.Forward(ctx, x)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this QuantGemma.
// Reuses [GemmaCache] — the cache structure is identical to the float model's; only the
// projections differ.
func (m *QuantGemma) NewCache() *GemmaCache {
	return &GemmaCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the √dim-scaled f32 embedding [1, dim] of a single token — the
// quantized twin of Gemma.embedOne: lookup from the dequantized table, then the same
// runtime ×√dim on the residual stream (the tied head keeps the unscaled bytes).
func (m *QuantGemma) embedOne(ctx *backend.Context, token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	x := embedRow(m.TokEmb, token, m.Config.Dim)
	return exec1(ctx, backend.OpMul, nil, x, m.embedScale())
}

// DecodeStep advances the quantized Gemma by one token at absolute position pos,
// appending its post-RoPE K/V to the cache and returning the next-token logits
// [1, vocab]. It mirrors Gemma.DecodeStep exactly — √dim-scaled single-token embedding,
// quantized projections, RoPE at PosOffset=pos, the single query attending to the whole
// cache without a causal mask, GeGLU FFN, tied head — so a KV-cache decode matches the
// full Forward up to f32 reassociation. Inference-only, like the rest of the type.
func (m *QuantGemma) DecodeStep(ctx *backend.Context, cache *GemmaCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	x, err := m.embedOne(ctx, token)
	if err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
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
		if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}, k); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false}, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		o, err := b.Wo.Forward(ctx, a)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
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
func (m *QuantGemma) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
// projections (attention, FFN, the tied head). Idempotent; call it when done with the
// model to release GPU memory promptly.
func (m *QuantGemma) Close() error {
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
