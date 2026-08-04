package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantStableLM is a [StableLM] whose projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized StableLM checkpoint, the stablelm
// sibling of [QuantLlama] / [QuantStarCoder2]. Its forward mirrors StableLM.Forward
// faithfully, all three StableLM departures included:
//
//   - full LayerNorm WITH bias (not RMSNorm): input_layernorm, post_attention_layernorm
//     and the final model.norm each carry a weight (γ) AND a bias (β), kept as f32
//     [nn.LayerNorm]s exactly as the float type does. β is NOT forced to zero (the
//     Cohere weight-only case) — dropping it moves the logits far off the golden.
//   - SEQUENTIAL residual (use_parallel_residual=False): the classic Llama two-norm
//     block, x = x + attention(input_layernorm(x)); x = x + FFN(post_attention_layernorm(x)).
//   - PARTIAL split-half rotary: only the first rotaryDim channels of each head rotate
//     ([partialRoPE], the same helper the float path uses — RotaryDim /
//     [StableLMConfig.rotaryDim]); the tail passes through unrotated.
//
// The attention projections are BIAS-FREE (use_qkv_bias=false, the default config the
// float type models — no quantProjBias needed anywhere), attention is causal GQA at the
// standard 1/√head_dim scale, the FFN is a bias-free SwiGLU ([nn.QuantSwiGLU]) and the
// LM head is UNTIED (lm_head.weight — required, the stablelm architecture has no
// tied-embedding fallback).
//
// Dtype discipline (the §B quant-bias lesson): the quantized residual stream is f32 —
// QuantLinear outputs f32 activations — so every small tensor that touches it is f32
// too: the LayerNorm γ/β pairs and the dequantized embedding table. Real
// llama.cpp-quantized stablelm files store all of these as F32 (1-D tensors are never
// block-quantized), so the GGUF path and [QuantizeStableLM] land on identical bytes.
// Inference-only: the quantized weights are frozen and bypass the tape.
//
// Load a llama.cpp-quantized checkpoint with [QuantStableLMFromGGUF], or quantize a
// float model with [QuantizeStableLM].
type QuantStableLM struct {
	Config StableLMConfig        // model geometry, shared with float StableLM (see StableLMConfig)
	TokEmb *tensor.Tensor        // [vocab, dim] f32 token embedding (lookup only)
	Blocks []*QuantStableLMBlock // the quantized StableLM blocks
	Norm   *nn.LayerNorm         // final pre-logits LayerNorm (f32 γ+β)
	Out    *nn.QuantLinear       // untied LM head (output.weight, required)
}

// QuantStableLMBlock is one sequential-residual QuantStableLM block: f32 full
// LayerNorms (γ+β) gating quantized bias-free attention projections and a quantized
// bias-free SwiGLU FFN. The QuantLinear weights carry the bulk of the bytes; the norm
// vectors stay f32, llama.cpp's on-disk convention for 1-D tensors.
type QuantStableLMBlock struct {
	InputNorm      *nn.LayerNorm   // input_layernorm (f32 γ+β), feeds attention
	PostAttnNorm   *nn.LayerNorm   // post_attention_layernorm (f32 γ+β), feeds the FFN
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (no bias — use_qkv_bias=false)
	FFN            *nn.QuantSwiGLU // quantized SwiGLU feed-forward (silu gate, no bias)
}

// QuantizeStableLM builds a QuantStableLM from a float StableLM by quantizing every
// 2-D projection (attn q/k/v/o, the SwiGLU gate/up/down, the untied head) to qt — the
// projections carry the bulk of the weights and compute, so this is where quantization
// pays off. Everything 1-D stays f32: the LayerNorm γ/β pairs (tiny, precision-sensitive,
// and — matching llama.cpp's on-disk convention — never block-quantized), plus the token
// embedding table (its lookup needs a float table; StableLM's head is untied, so unlike
// [QuantizeGemma] the table itself need not be quantized). This makes the result
// byte-comparable to [QuantStableLMFromGGUF] on a file that quantizes exactly the 2-D
// projections — the exact-anchor gate. Each projection's inner dimension must be a
// multiple of qt's block size (32 for Q8_0/Q4_0, 256 for the k-quants).
func QuantizeStableLM(m *StableLM, qt gguf.QuantType) (*QuantStableLM, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantStableLM{
		Config: m.Config,
		TokEmb: f32Clone(m.TokEmb),
		Norm:   f32LayerNorm(m.Norm),
	}
	var err error
	for _, b := range m.Blocks {
		qb := &QuantStableLMBlock{
			InputNorm:    f32LayerNorm(b.InputNorm),
			PostAttnNorm: f32LayerNorm(b.PostAttnNorm),
		}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStableLM Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStableLM Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStableLM Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStableLM Wo: %w", err)
		}
		gate, err := mkQ(b.FFN.Wgate)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStableLM ffn_gate: %w", err)
		}
		up, err := mkQ(b.FFN.Wup)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStableLM ffn_up: %w", err)
		}
		down, err := mkQ(b.FFN.Wdown)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStableLM ffn_down: %w", err)
		}
		qb.FFN = &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Out, err = mkQ(m.Out); err != nil {
		return nil, fmt.Errorf("nlp: QuantizeStableLM output: %w", err)
	}
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It
// mirrors StableLM.Forward exactly — embedding lookup (no positional embedding, no
// scale), then per block the sequential residual
//
//	x = x + attention(input_layernorm(x)); x = x + FFN(post_attention_layernorm(x))
//
// with bias-free quantized projections, PARTIAL split-half RoPE (only the first
// rotaryDim channels of each head rotate) and causal GQA, then the final LayerNorm and
// the quantized untied head — but every projection is a quantized in-kernel matmul, all
// activations f32.
func (m *QuantStableLM) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: StableLM prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
	rot := cfg.rotaryDim()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for _, b := range m.Blocks {
		// Attention sublayer: input_layernorm → bias-free q/k/v → partial RoPE →
		// causal GQA → bias-free o_proj → residual add.
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
		//perfscan:ignore PS6016 per-block RoPEAttrs literal, MHA/matmul-dominated block loop
		if q, err = partialRoPE(ctx, q, cfg.Heads, rot, backend.RoPEAttrs{Base: cfg.RopeBase}); err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016 per-block RoPEAttrs literal, MHA/matmul-dominated block loop
		if k, err = partialRoPE(ctx, k, kv, rot, backend.RoPEAttrs{Base: cfg.RopeBase}); err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 resource-only variadic exec1, per-block, MHA-dominated
		a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		if a, err = b.Wo.Forward(ctx, a); err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 resource-only variadic exec1 OpAdd, per-block negligible
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// FFN sublayer: post_attention_layernorm → SwiGLU → residual add.
		xn2, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := b.FFN.Forward(ctx, xn2)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 resource-only variadic exec1 OpAdd, per-block negligible
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this
// QuantStableLM. Reuses [StableLMCache] — the cache structure is identical to the
// float model's (post-partial-RoPE keys, raw values); only the projections differ.
func (m *QuantStableLM) NewCache() *StableLMCache {
	return &StableLMCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token (StableLM has no
// embedding scale).
func (m *QuantStableLM) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized StableLM by one token at absolute position pos,
// appending its post-partial-RoPE K and raw V to the cache and returning the next-token
// logits [1, vocab]. It mirrors StableLM.DecodeStep exactly — single-token embedding,
// bias-free quantized projections, partial RoPE at PosOffset=pos (the unrotated tail
// channels pass through untouched), the single query attending to the whole cache
// without a causal mask, the SwiGLU FFN, final LayerNorm, untied head — so a KV-cache
// decode matches the full Forward (same kernel sequence per row). Inference-only, like
// the rest of the type.
func (m *QuantStableLM) DecodeStep(ctx *backend.Context, cache *StableLMCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	rot := cfg.rotaryDim()
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
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
		//perfscan:ignore PS6016 per-block RoPEAttrs literal, decode dominated by projections
		if q, err = partialRoPE(ctx, q, cfg.Heads, rot, backend.RoPEAttrs{Base: cfg.RopeBase, PosOffset: pos}); err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016 per-block RoPEAttrs literal, decode dominated by projections
		if k, err = partialRoPE(ctx, k, kv, rot, backend.RoPEAttrs{Base: cfg.RopeBase, PosOffset: pos}); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query attends to all cached keys → no causal mask
		//perfscan:ignore PS6016,PS6017 per-block AttnAttrs literal, MHA-dominated decode | resource-only variadic exec1 OpMHA, per-block
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false}, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = b.Wo.Forward(ctx, a); err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 resource-only variadic exec1 OpAdd, per-block negligible
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		xn2, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := b.FFN.Forward(ctx, xn2)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 resource-only variadic exec1 OpAdd, per-block negligible
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on the
// quantized model, using the KV-cache (each step is one token, not a full re-forward),
// and returns prompt+new. The sampler s selects each token (Greedy() for deterministic
// argmax). Stops at the context limit. The same shape as [QuantLlama.Generate].
func (m *QuantStableLM) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
// projections (attention, FFN, the head). Idempotent; call it when done with the model
// to release GPU memory promptly.
func (m *QuantStableLM) Close() error {
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
