package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantStarCoder2 is a [StarCoder2] whose projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized StarCoder2 checkpoint, the starcoder2
// sibling of [QuantLlama] / [QuantGemma]. Its forward mirrors StarCoder2.Forward
// faithfully — full LayerNorm (γ AND β) around each sublayer, BIASED q/k/v/o
// projections, full NEOX split-half RoPE, grouped-query attention, the biased 2-layer
// GELU MLP (c_fc → GELU → c_proj, no gate), the final LayerNorm and the untied (or
// tied-fallback) LM head — but every 2-D projection is an in-kernel dequantized
// [nn.QuantLinear] that runs on the active accelerator when it supports the quant type
// (else the CPU fallback).
//
// Dtype discipline (the §B quant-bias lesson): the quantized residual stream is f32 —
// QuantLinear outputs f32 activations — so every small tensor that touches it is kept
// f32 too: the projection biases (added via OpAddBias right after each quantized
// matmul, in exactly the float forward's order), the LayerNorm γ/β pairs and the
// dequantized embedding table. Real llama.cpp-quantized files store all of these as
// F32 (1-D tensors are never block-quantized), so the GGUF path and [QuantizeStarCoder2]
// land on identical bytes. Inference-only: the quantized weights are frozen and bypass
// the tape.
//
// Load a llama.cpp-quantized checkpoint with [QuantStarCoder2FromGGUF], or quantize a
// float model with [QuantizeStarCoder2].
type QuantStarCoder2 struct {
	Config StarCoder2Config        // model geometry, shared with float StarCoder2 (see StarCoder2Config)
	TokEmb *tensor.Tensor          // [vocab, dim] f32 token embedding (lookup only)
	Blocks []*QuantStarCoder2Block // the quantized StarCoder2 blocks
	Norm   *nn.LayerNorm           // final pre-logits LayerNorm (f32 γ+β)
	Out    *nn.QuantLinear         // LM head (untied output.weight, or the tied token_embd bytes)
}

// QuantStarCoder2Block is one sequential-residual QuantStarCoder2 block: f32 full
// LayerNorms (γ+β) gating quantized BIASED attention projections and a quantized biased
// 2-layer GELU MLP. The QuantLinear weights carry the bulk of the bytes; the biases are
// separate f32 vectors added after each quantized matmul — llama.cpp's quantized
// starcoder2 files keep them F32 on disk for the same reason (1-D tensors are never
// block-quantized), so this split IS the on-disk structure, not an approximation.
type QuantStarCoder2Block struct {
	InputNorm      *nn.LayerNorm   // input_layernorm (f32 γ+β), feeds attention
	PostAttnNorm   *nn.LayerNorm   // post_attention_layernorm (f32 γ+β), feeds the MLP
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections
	Bq, Bk, Bv, Bo *tensor.Tensor  // their f32 biases (use_bias=True — always present)
	Wfc            *nn.QuantLinear // mlp.c_fc (dim → hidden), quantized
	Bfc            *tensor.Tensor  // c_fc bias (f32)
	Wproj          *nn.QuantLinear // mlp.c_proj (hidden → dim), quantized
	Bproj          *tensor.Tensor  // c_proj bias (f32)
}

// quantProjBias computes x·dequant(W)ᵀ + bias — the quantized twin of [projBias]: the
// QuantLinear matmul produces f32 activations and the f32 bias is added with the same
// OpAddBias the float path uses, keeping the operation order (and therefore the math)
// identical to StarCoder2's biased projections.
func quantProjBias(ctx *backend.Context, x *tensor.Tensor, w *nn.QuantLinear, bias *tensor.Tensor) (*tensor.Tensor, error) {
	y, err := w.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	return addBiasIf(ctx, y, bias)
}

// QuantizeStarCoder2 builds a QuantStarCoder2 from a float StarCoder2 by quantizing
// every 2-D projection (attn q/k/v/o, mlp c_fc/c_proj, the untied head) to qt — the
// projections carry the bulk of the weights and compute, so this is where quantization
// pays off. Everything 1-D stays f32: the LayerNorm γ/β pairs and ALL projection biases
// (tiny, precision-sensitive, and — matching llama.cpp's on-disk convention — never
// block-quantized), plus the token embedding table (its lookup needs a float table;
// StarCoder2's head is untied, so unlike [QuantizeGemma] the table itself need not be
// quantized). This makes the result byte-comparable to [QuantStarCoder2FromGGUF] on a
// file that quantizes exactly the 2-D projections — the exact-anchor gate. Each
// projection's inner dimension must be a multiple of qt's block size (32 for Q8_0/Q4_0,
// 256 for the k-quants).
func QuantizeStarCoder2(m *StarCoder2, qt gguf.QuantType) (*QuantStarCoder2, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantStarCoder2{
		Config: m.Config,
		TokEmb: f32Clone(m.TokEmb),
		Norm:   f32LayerNorm(m.Norm),
	}
	var err error
	for _, b := range m.Blocks {
		qb := &QuantStarCoder2Block{
			InputNorm:    f32LayerNorm(b.InputNorm),
			PostAttnNorm: f32LayerNorm(b.PostAttnNorm),
			Bq:           f32Clone(b.Bq), Bk: f32Clone(b.Bk), Bv: f32Clone(b.Bv), Bo: f32Clone(b.Bo),
			Bfc: f32Clone(b.Bfc), Bproj: f32Clone(b.Bproj),
		}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStarCoder2 Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStarCoder2 Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStarCoder2 Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStarCoder2 Wo: %w", err)
		}
		if qb.Wfc, err = mkQ(b.Wfc); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStarCoder2 c_fc: %w", err)
		}
		if qb.Wproj, err = mkQ(b.Wproj); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeStarCoder2 c_proj: %w", err)
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Out, err = mkQ(m.Out); err != nil {
		return nil, fmt.Errorf("nlp: QuantizeStarCoder2 output: %w", err)
	}
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It
// mirrors StarCoder2.Forward exactly — embedding lookup (no positional embedding, no
// scale), then per block the sequential residual
//
//	x = x + attention(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x))
//
// with biased quantized projections, full NEOX RoPE and causal GQA, then the final
// LayerNorm and the quantized head — but every projection is a quantized in-kernel
// matmul, all activations f32.
func (m *QuantStarCoder2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: StarCoder2 prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv})
	for _, b := range m.Blocks {
		// Attention sublayer: input_layernorm → biased q/k/v → full RoPE → causal GQA →
		// biased o_proj → residual add.
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := quantProjBias(ctx, xn, b.Wq, b.Bq)
		if err != nil {
			return nil, err
		}
		k, err := quantProjBias(ctx, xn, b.Wk, b.Bk)
		if err != nil {
			return nil, err
		}
		v, err := quantProjBias(ctx, xn, b.Wv, b.Bv)
		if err != nil {
			return nil, err
		}
		if q, err = exec1a(ctx, backend.OpRoPE, qRoPE, q); err != nil {
			return nil, err
		}
		if k, err = exec1a(ctx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		a, err := exec3(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		if a, err = quantProjBias(ctx, a, b.Wo, b.Bo); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// MLP sublayer: post_attention_layernorm → biased GELU 2-layer → residual add.
		xn2, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, xn2)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// mlp runs the biased 2-layer GELU feed-forward on the normalized input xn [seq, dim]:
// c_fc → GELU → c_proj, the quantized twin of StarCoder2.mlp. Like the float pipeline
// it uses the exact erf GELU (OpGELU), so a checkpoint trained with the tanh
// approximation carries the same small, documented residual as the float path.
func (m *QuantStarCoder2) mlp(ctx *backend.Context, b *QuantStarCoder2Block, xn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := quantProjBias(ctx, xn, b.Wfc, b.Bfc)
	if err != nil {
		return nil, err
	}
	if h, err = exec1a(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return quantProjBias(ctx, h, b.Wproj, b.Bproj)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this
// QuantStarCoder2. Reuses [StarCoder2Cache] — the cache structure is identical to the
// float model's; only the projections differ.
func (m *QuantStarCoder2) NewCache() *StarCoder2Cache {
	return &StarCoder2Cache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token (StarCoder2 has no
// embedding scale).
func (m *QuantStarCoder2) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized StarCoder2 by one token at absolute position pos,
// appending its post-bias, post-RoPE K/V to the cache and returning the next-token
// logits [1, vocab]. It mirrors StarCoder2.DecodeStep exactly — single-token embedding,
// biased quantized projections, full RoPE at PosOffset=pos, the single query attending
// to the whole cache without a causal mask, biased GELU MLP, final LayerNorm, head — so
// a KV-cache decode matches the full Forward (same kernel sequence per row).
// Inference-only, like the rest of the type.
func (m *QuantStarCoder2) DecodeStep(ctx *backend.Context, cache *StarCoder2Cache, token, pos int) (*tensor.Tensor, error) {
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
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := quantProjBias(ctx, xn, b.Wq, b.Bq)
		if err != nil {
			return nil, err
		}
		k, err := quantProjBias(ctx, xn, b.Wk, b.Bk)
		if err != nil {
			return nil, err
		}
		v, err := quantProjBias(ctx, xn, b.Wv, b.Bv)
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
		if a, err = quantProjBias(ctx, a, b.Wo, b.Bo); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		xn2, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, xn2)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
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
func (m *QuantStarCoder2) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
// projections (attention, MLP, the head). Idempotent; call it when done with the model
// to release GPU memory promptly.
func (m *QuantStarCoder2) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, b := range m.Blocks {
		for _, l := range []*nn.QuantLinear{b.Wq, b.Wk, b.Wv, b.Wo, b.Wfc, b.Wproj} {
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

// f32LayerNorm returns a LayerNorm with f32 copies of γ AND β, so it composes with the
// f32 activations of the quantized pipeline — the full-LayerNorm counterpart of
// [f32RMSNorm], used by [QuantizeStarCoder2].
func f32LayerNorm(n *nn.LayerNorm) *nn.LayerNorm {
	return &nn.LayerNorm{Gamma: f32Clone(n.Gamma), Beta: f32Clone(n.Beta), Eps: n.Eps}
}
