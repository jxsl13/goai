package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantCohere is a [Cohere] (Command-R) whose projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized command-r checkpoint, the cohere
// sibling of [QuantLlama] / [QuantFalcon]. Its forward mirrors Cohere.Forward
// faithfully, all four Command-R departures included:
//
//   - PARALLEL residual: ONE weight-only LayerNorm (input_layernorm) feeds BOTH the
//     attention and the SwiGLU FFN, and both sublayer outputs are summed onto the raw
//     residual — xn = input_layernorm(x); x = x + attention(xn) + FFN(xn). There is no
//     post-norm and no separate FFN norm.
//   - weight-only mean-centered LayerNorm: every norm is an [nn.LayerNorm] whose β is a
//     ZERO vector (the float type's exact representation of CohereLayerNorm), kept f32
//     here so it composes with the quantized pipeline's f32 activations.
//   - INTERLEAVED RoPE via weight permutation: the q/k projection rows carry the
//     interleaved→split-half permutation (applied by [QuantizeCohere] via the float
//     model's already-permuted weights, or by [QuantCohereFromGGUF] directly on the
//     quantized bytes), so GoAI's split-half OpRoPE reproduces Cohere's interleaved
//     rotary exactly.
//   - logit_scale: the final logits are multiplied by config.logit_scale (e.g. 0.0625)
//     after the head, at the activations' f32 precision — llama.cpp's ggml_scale is
//     likewise a float multiply. A zero/absent scale means unscaled ([CohereConfig]'s
//     0 → 1 convention).
//
// The tied head is structural, like [QuantGemma]'s: the command-r architecture has no
// output.weight, so Out wraps the SAME quantized token-embedding bytes as a QuantLinear
// (logits = hidden · dequant(table)ᵀ) while TokEmb holds the dequantized f32 table those
// bytes decode to, used only for the embedding lookup. Both views come from one on-disk
// tensor, exactly as in a real quantized command-r GGUF. Activations are f32 —
// quantized-inference precision, which is also what engages the f32-only GPU kernels.
// Inference-only: the quantized weights are frozen and bypass the tape.
//
// Load a llama.cpp-quantized checkpoint with [QuantCohereFromGGUF], or quantize a float
// model with [QuantizeCohere].
type QuantCohere struct {
	Config CohereConfig        // model geometry, shared with float Cohere (see CohereConfig)
	TokEmb *tensor.Tensor      // [vocab, dim] f32 DEQUANTIZED embedding table (lookup only)
	Blocks []*QuantCohereBlock // the quantized parallel-residual Cohere blocks
	Norm   *nn.LayerNorm       // final pre-logits LayerNorm (f32 γ, zero β — weight-only)
	Out    *nn.QuantLinear     // tied LM head: the quantized token_embd bytes, In=dim Out=vocab
}

// QuantCohereBlock is one parallel-residual QuantCohere block: the single f32
// weight-only LayerNorm read by BOTH sublayers, quantized bias-free attention
// projections (q/k rows in the split-half RoPE layout) and a quantized bias-free
// SwiGLU. The QuantLinear weights carry the bulk of the bytes; the norm gain stays f32,
// llama.cpp's on-disk convention for 1-D tensors.
type QuantCohereBlock struct {
	InputNorm      *nn.LayerNorm   // input_layernorm (f32 γ, zero β), feeds attention AND FFN
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (no bias)
	FFN            *nn.QuantSwiGLU // quantized SwiGLU feed-forward (no bias)
}

// QuantizeCohere builds a QuantCohere from a float Cohere by quantizing every 2-D
// projection (attn q/k/v/o, the SwiGLU gate/up/down, the tied embedding table) to qt —
// the projections carry the bulk of the weights and compute, so this is where
// quantization pays off. Everything 1-D stays f32: the LayerNorm γ (and its zero β) —
// tiny, precision-sensitive, and, matching llama.cpp's on-disk convention, never
// block-quantized.
//
// Like [QuantizeGemma] — and unlike the untied-head twins — the token embedding IS
// quantized: Command-R's LM head is the embedding table, so the [vocab, dim] table
// (already the head's [out, in] layout) is encoded once and the SAME bytes serve as the
// head (Out), while TokEmb becomes the table those bytes dequantize to. Keeping the
// original float table for lookups would desynchronize the two tied views; using the
// dequantized one reproduces exactly what [QuantCohereFromGGUF] yields from a file
// quantized from the same weights — the exact-anchor gate. A Cohere whose Out is NOT
// the transpose of TokEmb (an untied lm_head loaded by [CohereFromHF]'s tolerance)
// cannot be represented and is rejected. The q/k weights are quantized as stored —
// i.e. AFTER the float loader's interleaved→split-half row permutation — so the bytes
// are directly comparable to [QuantCohereFromGGUF]'s byte-level permutation of the
// on-disk interleaved rows (quantize∘permute = permute∘quantize: ggml blocks are
// row-granular, so a row permutation commutes with per-row quantization). Each
// projection's inner dimension must be a multiple of qt's block size (32 for
// Q8_0/Q4_0, 256 for the k-quants) — for the tied head that means Dim itself.
func QuantizeCohere(m *Cohere, qt gguf.QuantType) (*QuantCohere, error) {
	if !equalsTransposed(m.Out, m.TokEmb) {
		return nil, fmt.Errorf("nlp: QuantizeCohere needs a tied head (Out = TokEmbᵀ); the command-r architecture has no untied output projection")
	}
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	// Tied head: the [vocab, dim] table is already [out, in] — no transpose. One
	// encoding serves both views (lookup table + head), the QuantizeGemma pattern.
	vocab, dim := m.TokEmb.Shape()[0], m.TokEmb.Shape()[1]
	headBytes, err := gguf.Quantize(m.TokEmb, qt)
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeCohere token embedding: %w", err)
	}
	emb, err := (gguf.QuantTensor{Data: headBytes, GGType: uint32(qt), Shape: tensor.Shape{vocab, dim}}).Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeCohere token embedding: %w", err)
	}
	q := &QuantCohere{
		Config: m.Config,
		TokEmb: emb,
		Norm:   f32LayerNorm(m.Norm),
		Out:    &nn.QuantLinear{Weight: headBytes, QT: qt, In: dim, Out: vocab},
	}
	for _, b := range m.Blocks {
		qb := &QuantCohereBlock{InputNorm: f32LayerNorm(b.InputNorm)}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeCohere Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeCohere Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeCohere Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeCohere Wo: %w", err)
		}
		gate, err := mkQ(b.FFN.Wgate)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeCohere ffn_gate: %w", err)
		}
		up, err := mkQ(b.FFN.Wup)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeCohere ffn_up: %w", err)
		}
		down, err := mkQ(b.FFN.Wdown)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeCohere ffn_down: %w", err)
		}
		qb.FFN = &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down}
		q.Blocks = append(q.Blocks, qb)
	}
	return q, nil
}

// scaleLogits multiplies the raw logits by config.logit_scale — the quantized twin of
// [Cohere.scaleLogits], at the quantized pipeline's f32 activation precision (the same
// float multiply llama.cpp's ggml_scale applies; the float pipeline's f64 scalar would
// force a mixed-dtype broadcast onto the f32 logits). A scale of 1 (or the 0-means-1
// convention) is returned unchanged.
func (m *QuantCohere) scaleLogits(ctx *backend.Context, logits *tensor.Tensor) (*tensor.Tensor, error) {
	s := m.Config.logitScale()
	if s == 1 {
		return logits, nil
	}
	scale := tensor.New(tensor.F32, tensor.Shape{})
	scale.Storage().F32()[0] = float32(s)
	return exec2(ctx, backend.OpMul, nil, logits, scale)
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]
// (already multiplied by logit_scale). It mirrors Cohere.Forward exactly — embedding
// lookup (no positional embedding, no scale), then per block the parallel residual
//
//	xn = input_layernorm(x); x = x + attention(xn) + FFN(xn)
//
// with bias-free quantized projections, split-half OpRoPE on the permuted q/k rows
// (Cohere's interleaved rotary) and causal GQA, then the final LayerNorm, the tied
// quantized head and the logit scale — but every projection is a quantized in-kernel
// matmul, all activations f32.
func (m *QuantCohere) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: Cohere prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for _, b := range m.Blocks {
		// Parallel residual: ONE norm feeds both sublayers; their outputs are summed
		// onto the raw residual — the float hiddenCapture's exact graph and add order.
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.attention(ctx, b, xn, attn)
		if err != nil {
			return nil, err
		}
		f, err := b.FFN.Forward(ctx, xn)
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
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	logits, err := m.Out.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	return m.scaleLogits(ctx, logits)
}

// attention computes the quantized Command-R attention over the normalized input xn
// [seq, dim]: quantized q/k/v projections, split-half OpRoPE (which reproduces Cohere's
// interleaved rotary because the q/k rows are permuted), causal GQA via OpMHA and the
// quantized o_proj — the quantized twin of [Cohere.attention], same kernel order (no
// QK-norm in Cohere v1).
func (m *QuantCohere) attention(ctx *backend.Context, b *QuantCohereBlock, xn *tensor.Tensor, attn backend.AttnAttrs) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	q, err := b.Wq.Forward(ctx, xn)
	if err != nil {
		return nil, err
	}
	k, err := b.Wk.Forward(ctx, xn)
	if err != nil {
		return nil, err
	}
	v, err := b.Wv.Forward(ctx, xn)
	if err != nil {
		return nil, err
	}
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, k); err != nil {
		return nil, err
	}
	a, err := exec3(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return b.Wo.Forward(ctx, a)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this QuantCohere.
// Reuses [CohereCache] — the cache structure is identical to the float model's
// (post-RoPE keys, raw values); only the projections differ.
func (m *QuantCohere) NewCache() *CohereCache {
	return &CohereCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token from the
// dequantized table (Command-R has no embedding scale; the tied head keeps using the
// quantized bytes).
func (m *QuantCohere) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized Command-R by one token at absolute position pos,
// appending its post-RoPE K and raw V to the cache and returning the next-token logits
// [1, vocab] (already scaled by logit_scale). It mirrors Cohere.DecodeStep exactly —
// single-token embedding, the ONE input_layernorm feeding both sublayers, quantized
// projections, split-half RoPE at PosOffset=pos on the permuted q/k, the single query
// attending to the whole cache without a causal mask, both sublayer outputs summed onto
// the raw residual, final LayerNorm, tied head, logit scale — so a KV-cache decode
// matches the full Forward (same kernel sequence per row). Inference-only, like the
// rest of the type.
func (m *QuantCohere) DecodeStep(ctx *backend.Context, cache *CohereCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos})
	attnA := backend.Attrs(backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false})
	for l, b := range m.Blocks {
		// Parallel residual: ONE norm feeds both sublayers.
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := b.Wq.Forward(ctx, xn)
		if err != nil {
			return nil, err
		}
		k, err := b.Wk.Forward(ctx, xn)
		if err != nil {
			return nil, err
		}
		v, err := b.Wv.Forward(ctx, xn)
		if err != nil {
			return nil, err
		}
		if q, err = exec1a(ctx, backend.OpRoPE, qRoPE, q); err != nil {
			return nil, err
		}
		if k, err = exec1a(ctx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query attends to all cached keys → no causal mask
		a, err := exec3(ctx, backend.OpMHA, attnA, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = b.Wo.Forward(ctx, a); err != nil {
			return nil, err
		}
		f, err := b.FFN.Forward(ctx, xn)
		if err != nil {
			return nil, err
		}
		// Sum both sublayer outputs onto the raw residual — the float DecodeStep's order.
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	logits, err := m.Out.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	return m.scaleLogits(ctx, logits)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on the
// quantized model, using the KV-cache (each step is one token, not a full re-forward),
// and returns prompt+new. The sampler s selects each token (Greedy() for deterministic
// argmax). Stops at the context limit. The same shape as [QuantLlama.Generate].
func (m *QuantCohere) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
func (m *QuantCohere) Close() error {
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
