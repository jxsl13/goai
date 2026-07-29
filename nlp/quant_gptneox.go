package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantGPTNeoX is a [GPTNeoX] whose projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized GPT-NeoX / Pythia checkpoint, the
// gptneox sibling of [QuantStarCoder2]. Its forward mirrors GPTNeoX.Forward
// faithfully — the PARALLEL residual (input_layernorm feeds attention,
// post_attention_layernorm feeds the MLP, BOTH read the same raw residual and both
// outputs are summed onto it), full LayerNorm (γ AND β) everywhere, BIASED q/k/v/o
// projections, PARTIAL split-half RoPE over RotaryDim channels per head, MHA (no
// GQA), the biased 2-layer GELU MLP (dense_h_to_4h → GELU → dense_4h_to_h) and the
// UNTIED LM head — but every 2-D projection is an in-kernel dequantized
// [nn.QuantLinear] that runs on the active accelerator when it supports the quant
// type (else the CPU fallback).
//
// Dtype discipline (the §B quant-bias lesson): the quantized residual stream is f32 —
// QuantLinear outputs f32 activations — so every small tensor that touches it is kept
// f32 too: the projection biases (added via OpAddBias right after each quantized
// matmul, in exactly the float forward's order), the LayerNorm γ/β pairs and the
// dequantized embedding table. Real llama.cpp-quantized files store all of these as
// F32 (1-D tensors are never block-quantized), so the GGUF path and [QuantizeGPTNeoX]
// land on identical bytes. Inference-only: the quantized weights are frozen and bypass
// the tape.
//
// Load a llama.cpp-quantized checkpoint with [QuantGPTNeoXFromGGUF], or quantize a
// float model with [QuantizeGPTNeoX].
type QuantGPTNeoX struct {
	Config    GPTNeoXConfig        // model geometry, shared with float GPT-NeoX (see GPTNeoXConfig)
	TokEmb    *tensor.Tensor       // [vocab, dim] f32 token embedding (lookup only)
	Blocks    []*QuantGPTNeoXBlock // the quantized GPT-NeoX blocks
	FinalNorm *nn.LayerNorm        // final_layer_norm (f32 γ+β)
	Out       *nn.QuantLinear      // LM head (embed_out — untied, required)
}

// QuantGPTNeoXBlock is one PARALLEL-residual QuantGPTNeoX block: two f32 full
// LayerNorms (γ+β) reading the SAME raw residual — InputNorm gates the quantized
// biased attention, PostAttnNorm gates the quantized biased GELU MLP — with both
// sublayer outputs summed onto the residual. The QuantLinear weights carry the bulk
// of the bytes; the biases are separate f32 vectors added after each quantized
// matmul — llama.cpp's quantized gptneox files keep them F32 on disk for the same
// reason (1-D tensors are never block-quantized), so this split IS the on-disk
// structure, not an approximation.
type QuantGPTNeoXBlock struct {
	InputNorm      *nn.LayerNorm   // input_layernorm (f32 γ+β), feeds attention
	PostAttnNorm   *nn.LayerNorm   // post_attention_layernorm (f32 γ+β), feeds the MLP
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (query_key_value split + dense)
	Bq, Bk, Bv, Bo *tensor.Tensor  // their f32 biases (every GPT-NeoX projection is biased)
	Wh             *nn.QuantLinear // mlp.dense_h_to_4h (dim → hidden), quantized
	Bh             *tensor.Tensor  // its bias (f32)
	Wout           *nn.QuantLinear // mlp.dense_4h_to_h (hidden → dim), quantized
	Bout           *tensor.Tensor  // its bias (f32)
}

// QuantizeGPTNeoX builds a QuantGPTNeoX from a float GPTNeoX by quantizing every 2-D
// projection (attn q/k/v/o, the two MLP denses, the untied head) to qt — the
// projections carry the bulk of the weights and compute, so this is where
// quantization pays off. Everything 1-D stays f32: the LayerNorm γ/β pairs and ALL
// projection biases (tiny, precision-sensitive, and — matching llama.cpp's on-disk
// convention — never block-quantized), plus the token embedding table (its lookup
// needs a float table; GPT-NeoX's head is untied, so the table itself need not be
// quantized). This makes the result byte-comparable to [QuantGPTNeoXFromGGUF] on a
// file that quantizes exactly the 2-D projections — the exact-anchor gate. Each
// projection's inner dimension must be a multiple of qt's block size (32 for
// Q8_0/Q4_0, 256 for the k-quants).
func QuantizeGPTNeoX(m *GPTNeoX, qt gguf.QuantType) (*QuantGPTNeoX, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantGPTNeoX{
		Config:    m.Config,
		TokEmb:    f32Clone(m.TokEmb),
		FinalNorm: f32LayerNorm(m.FinalNorm),
	}
	var err error
	for _, b := range m.Blocks {
		qb := &QuantGPTNeoXBlock{
			InputNorm:    f32LayerNorm(b.InputNorm),
			PostAttnNorm: f32LayerNorm(b.PostAttnNorm),
			Bq:           f32Clone(b.Bq), Bk: f32Clone(b.Bk), Bv: f32Clone(b.Bv), Bo: f32Clone(b.Bo),
			Bh: f32Clone(b.Bh), Bout: f32Clone(b.Bout),
		}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGPTNeoX Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGPTNeoX Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGPTNeoX Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGPTNeoX Wo: %w", err)
		}
		if qb.Wh, err = mkQ(b.Wh); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGPTNeoX dense_h_to_4h: %w", err)
		}
		if qb.Wout, err = mkQ(b.Wout); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeGPTNeoX dense_4h_to_h: %w", err)
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Out, err = mkQ(m.Out); err != nil {
		return nil, fmt.Errorf("nlp: QuantizeGPTNeoX output: %w", err)
	}
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab].
// It mirrors GPTNeoX.Forward exactly — embedding lookup (no positional embedding, no
// scale), then per block the PARALLEL residual
//
//	x = x + attention(input_layernorm(x)) + mlp(post_attention_layernorm(x))
//
// with biased quantized projections, PARTIAL split-half RoPE and causal MHA, then the
// final LayerNorm and the quantized untied head — but every projection is a quantized
// in-kernel matmul, all activations f32. The sublayer order (attention computed
// first, both sums applied after the MLP) matches the float path kernel-for-kernel,
// which is what makes the exact-anchor gate byte-comparable.
func (m *QuantGPTNeoX) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: GPTNeoX prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
		// Parallel residual: input_layernorm feeds attention, post_attention_layernorm
		// feeds the MLP, and both outputs are summed onto the SAME raw residual x.
		an, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.attention(ctx, b, an, attn, 0)
		if err != nil {
			return nil, err
		}
		fn, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, fn)
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

// attention computes GPT-NeoX multi-head attention over the normalized input an
// [seq, dim] with quantized projections: biased q/k/v, the PARTIAL split-half RoPE at
// position offset pos over RotaryDim channels per head (the tail passes through
// unrotated), causal MHA and the biased output dense — the quantized twin of
// [GPTNeoX.attention].
func (m *QuantGPTNeoX) attention(ctx *backend.Context, b *QuantGPTNeoXBlock, an *tensor.Tensor, attn backend.AttnAttrs, pos int) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	rot := cfg.rotaryDim()
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, PosOffset: pos}
	q, err := quantProjBias(ctx, an, b.Wq, b.Bq)
	if err != nil {
		return nil, err
	}
	k, err := quantProjBias(ctx, an, b.Wk, b.Bk)
	if err != nil {
		return nil, err
	}
	v, err := quantProjBias(ctx, an, b.Wv, b.Bv)
	if err != nil {
		return nil, err
	}
	if q, err = partialRoPE(ctx, q, cfg.Heads, rot, rope); err != nil {
		return nil, err
	}
	if k, err = partialRoPE(ctx, k, kv, rot, rope); err != nil {
		return nil, err
	}
	a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return quantProjBias(ctx, a, b.Wo, b.Bo)
}

// mlp runs the biased 2-layer GELU feed-forward on the normalized input fn
// [seq, dim]: dense_h_to_4h → exact-erf GELU → dense_4h_to_h, the quantized twin of
// [GPTNeoX.mlp].
func (m *QuantGPTNeoX) mlp(ctx *backend.Context, b *QuantGPTNeoXBlock, fn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := quantProjBias(ctx, fn, b.Wh, b.Bh)
	if err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return quantProjBias(ctx, h, b.Wout, b.Bout)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this
// QuantGPTNeoX. Reuses [GPTNeoXCache] — the cache structure is identical to the float
// model's; only the projections differ.
func (m *QuantGPTNeoX) NewCache() *GPTNeoXCache {
	return &GPTNeoXCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token (GPT-NeoX has no
// embedding scale).
func (m *QuantGPTNeoX) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized GPT-NeoX by one token at absolute position pos,
// appending its post-bias, post-partial-RoPE K/V to the cache and returning the
// next-token logits [1, vocab]. It mirrors GPTNeoX.DecodeStep exactly — single-token
// embedding, the parallel residual (attention from input_layernorm, MLP from
// post_attention_layernorm, both summed onto the raw residual), partial RoPE at
// PosOffset=pos, the single query attending to the whole cache without a causal mask,
// final LayerNorm, head — so a KV-cache decode matches the full Forward (same kernel
// sequence per row). Inference-only, like the rest of the type.
func (m *QuantGPTNeoX) DecodeStep(ctx *backend.Context, cache *GPTNeoXCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	rot := cfg.rotaryDim()
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, PosOffset: pos}
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	attnA := backend.Attrs(backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false})
	for l, b := range m.Blocks {
		// input_layernorm feeds attention; post_attention_layernorm feeds the MLP; both
		// read the same raw residual x.
		an, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := quantProjBias(ctx, an, b.Wq, b.Bq)
		if err != nil {
			return nil, err
		}
		k, err := quantProjBias(ctx, an, b.Wk, b.Bk)
		if err != nil {
			return nil, err
		}
		v, err := quantProjBias(ctx, an, b.Wv, b.Bv)
		if err != nil {
			return nil, err
		}
		if q, err = partialRoPE(ctx, q, cfg.Heads, rot, rope); err != nil {
			return nil, err
		}
		if k, err = partialRoPE(ctx, k, kv, rot, rope); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query attends to all cached keys → no causal mask
		a, err := exec3(ctx, backend.OpMHA, attnA, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = quantProjBias(ctx, a, b.Wo, b.Bo); err != nil {
			return nil, err
		}
		fn, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, fn)
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
func (m *QuantGPTNeoX) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
func (m *QuantGPTNeoX) Close() error {
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
