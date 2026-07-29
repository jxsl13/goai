package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantFalcon is a [Falcon] whose projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized Falcon-7B checkpoint, the falcon
// sibling of [QuantGPTNeoX] / [QuantStarCoder2]. Its forward mirrors Falcon.Forward
// faithfully — the SINGLE-NORM PARALLEL residual (ONE input_layernorm feeds BOTH the
// attention and the MLP, both sublayer outputs summed onto the raw residual),
// multi-query attention (all query heads share ONE key and ONE value head,
// KVHeads=1), full split-half NEOX RoPE, the bias-free 2-layer GELU MLP and the
// untied (or tied-fallback) LM head — but every 2-D projection is an in-kernel
// dequantized [nn.QuantLinear] that runs on the active accelerator when it supports
// the quant type (else the CPU fallback).
//
// Dtype discipline (the §B quant-bias lesson): the quantized residual stream is f32 —
// QuantLinear outputs f32 activations — so every small tensor that touches it is kept
// f32 too: the LayerNorm γ/β pairs (Falcon's LINEAR layers are bias-free, but its
// LayerNorms always carry β) and the dequantized embedding table. Real
// llama.cpp-quantized files store these as F32 (1-D tensors are never
// block-quantized), so the GGUF path and [QuantizeFalcon] land on identical bytes.
// Inference-only: the quantized weights are frozen and bypass the tape.
//
// Load a llama.cpp-quantized checkpoint with [QuantFalconFromGGUF], or quantize a
// float model with [QuantizeFalcon].
type QuantFalcon struct {
	Config    FalconConfig        // model geometry, shared with float Falcon (see FalconConfig)
	TokEmb    *tensor.Tensor      // [vocab, dim] f32 token embedding (lookup only)
	Blocks    []*QuantFalconBlock // the quantized Falcon blocks
	FinalNorm *nn.LayerNorm       // ln_f (f32 γ+β)
	Out       *nn.QuantLinear     // LM head (untied output.weight, or the tied token_embd bytes)
}

// QuantFalconBlock is one SINGLE-NORM parallel-residual QuantFalcon block: ONE f32
// full LayerNorm (γ+β) whose output feeds BOTH the quantized bias-free attention and
// the quantized bias-free GELU MLP, with both sublayer outputs summed onto the raw
// residual. There is no post-norm, no separate MLP norm and no projection bias
// anywhere (config.bias=False); the MQA split gives Wk/Wv a single head each.
type QuantFalconBlock struct {
	InputNorm      *nn.LayerNorm   // input_layernorm (f32 γ+β), feeds attention AND MLP
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (bias-free); Wk/Wv single-head (MQA)
	Wh             *nn.QuantLinear // mlp.dense_h_to_4h (dim → hidden), quantized, bias-free
	Wout           *nn.QuantLinear // mlp.dense_4h_to_h (hidden → dim), quantized, bias-free
}

// QuantizeFalcon builds a QuantFalcon from a float Falcon by quantizing every 2-D
// projection (attn q/k/v/o, the two MLP denses, the head) to qt — the projections
// carry the bulk of the weights and compute, so this is where quantization pays off.
// Everything 1-D stays f32: the LayerNorm γ/β pairs (tiny, precision-sensitive, and —
// matching llama.cpp's on-disk convention — never block-quantized), plus the token
// embedding table (its lookup needs a float table). This makes the result
// byte-comparable to [QuantFalconFromGGUF] on a file that quantizes exactly the 2-D
// projections — the exact-anchor gate. Each projection's inner dimension must be a
// multiple of qt's block size (32 for Q8_0/Q4_0, 256 for the k-quants).
func QuantizeFalcon(m *Falcon, qt gguf.QuantType) (*QuantFalcon, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantFalcon{
		Config:    m.Config,
		TokEmb:    f32Clone(m.TokEmb),
		FinalNorm: f32LayerNorm(m.FinalNorm),
	}
	var err error
	for _, b := range m.Blocks {
		qb := &QuantFalconBlock{InputNorm: f32LayerNorm(b.InputNorm)}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeFalcon Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeFalcon Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeFalcon Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeFalcon Wo: %w", err)
		}
		if qb.Wh, err = mkQ(b.Wh); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeFalcon dense_h_to_4h: %w", err)
		}
		if qb.Wout, err = mkQ(b.Wout); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeFalcon dense_4h_to_h: %w", err)
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Out, err = mkQ(m.Out); err != nil {
		return nil, fmt.Errorf("nlp: QuantizeFalcon output: %w", err)
	}
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab].
// It mirrors Falcon.Forward exactly — embedding lookup (no positional embedding, no
// scale), then per block the SINGLE-NORM parallel residual
//
//	xn = input_layernorm(x); x = x + attention(xn) + mlp(xn)
//
// with bias-free quantized projections, full NEOX RoPE and causal MQA (KVHeads=1),
// then the final LayerNorm and the quantized head — but every projection is a
// quantized in-kernel matmul, all activations f32. The sublayer order (attention
// computed first, both sums applied after the MLP) matches the float path
// kernel-for-kernel, which is what makes the exact-anchor gate byte-comparable.
func (m *QuantFalcon) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: Falcon prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: cfg.kvHeads(), Causal: true}
	for _, b := range m.Blocks {
		// Single-norm parallel residual: ONE norm feeds both sublayers; their outputs
		// are summed onto the raw residual. xn = input_layernorm(x); x = x + attn(xn) + mlp(xn).
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.attention(ctx, b, xn, attn, 0)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, xn)
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
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// attention computes Falcon's multi-query attention over the normalized input xn
// [seq, dim] with quantized bias-free projections: q/k/v — k and v have ONE head each
// (MQA) — full split-half RoPE at position offset pos, causal MQA via OpMHA
// (KVHeads=1) and the bias-free output dense — the quantized twin of
// [Falcon.attention].
func (m *QuantFalcon) attention(ctx *backend.Context, b *QuantFalconBlock, xn *tensor.Tensor, attn backend.AttnAttrs, pos int) (*tensor.Tensor, error) {
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
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.ropeBase(), Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.ropeBase(), Heads: kv, PosOffset: pos}, k); err != nil {
		return nil, err
	}
	a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return b.Wo.Forward(ctx, a)
}

// mlp runs the bias-free 2-layer GELU feed-forward on the normalized input xn
// [seq, dim]: dense_h_to_4h → exact-erf GELU → dense_4h_to_h, the quantized twin of
// [Falcon.mlp].
func (m *QuantFalcon) mlp(ctx *backend.Context, b *QuantFalconBlock, xn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := b.Wh.Forward(ctx, xn)
	if err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return b.Wout.Forward(ctx, h)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this
// QuantFalcon. Reuses [FalconCache] — the cache structure is identical to the float
// model's (single-head MQA k/v per block); only the projections differ.
func (m *QuantFalcon) NewCache() *FalconCache {
	return &FalconCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token (Falcon has no
// embedding scale).
func (m *QuantFalcon) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized Falcon by one token at absolute position pos,
// appending its post-RoPE single-head K and raw single-head V (MQA) to the cache and
// returning the next-token logits [1, vocab]. It mirrors Falcon.DecodeStep exactly —
// single-token embedding, the SINGLE norm feeding both sublayers, full RoPE at
// PosOffset=pos, the single query attending to the whole cache without a causal mask,
// the bias-free GELU MLP on the same normed input, both outputs summed onto the raw
// residual, final LayerNorm, head — so a KV-cache decode matches the full Forward
// (same kernel sequence per row). Inference-only, like the rest of the type.
func (m *QuantFalcon) DecodeStep(ctx *backend.Context, cache *FalconCache, token, pos int) (*tensor.Tensor, error) {
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
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.ropeBase(), Heads: cfg.Heads, PosOffset: pos})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.ropeBase(), Heads: kv, PosOffset: pos})
	attnA := backend.Attrs(backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false})
	for l, b := range m.Blocks {
		// Single-norm parallel residual: ONE norm feeds both sublayers.
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
		// Full split-half RoPE at the token's absolute position, then append k,v.
		if q, err = exec1a(ctx, backend.OpRoPE, qRoPE, q); err != nil {
			return nil, err
		}
		if k, err = exec1a(ctx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query attends all cached keys → no causal mask; MQA via KVHeads=1
		a, err := exec3(ctx, backend.OpMHA, attnA, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = b.Wo.Forward(ctx, a); err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, xn)
		if err != nil {
			return nil, err
		}
		// Sum both sublayer outputs onto the raw residual.
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
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
// quantized model, using the KV-cache (each step is one token, not a full
// re-forward), and returns prompt+new. The sampler s selects each token (Greedy() for
// deterministic argmax). Stops at the context limit. The same shape as
// [QuantStarCoder2.Generate].
func (m *QuantFalcon) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
// projections (attention, MLP, the head). Idempotent; call it when done with the
// model to release GPU memory promptly.
func (m *QuantFalcon) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, b := range m.Blocks {
		for _, l := range []*nn.QuantLinear{b.Wq, b.Wk, b.Wv, b.Wo, b.Wh, b.Wout} {
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
