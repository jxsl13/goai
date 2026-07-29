package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantNemotron is a [Nemotron] whose projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized Nemotron checkpoint, the nemotron
// sibling of [QuantLlama] / [QuantStarCoder2]. Its forward mirrors Nemotron.Forward
// faithfully, all three Nemotron departures included:
//
//   - LayerNorm1P norms: input_layernorm, post_attention_layernorm and the final norm
//     are mean-centered LayerNorms whose γ is the FOLDED (1 + hf_weight) gain with a
//     bias β alongside. The quantized twin carries them EXACTLY as the float type
//     does — γ already folded, β present — as f32 [nn.LayerNorm]s; no fold happens at
//     forward time (and none at GGUF load time either: the converter stores the
//     pre-folded gain on disk, see [QuantNemotronFromGGUF]).
//   - ReLU² MLP, NO gate, NO bias: down(relu²(up(x))) with relu²(y) = relu(y)² —
//     computed as u = up(x); r = ReLU(u); out = down(r ⊙ r), the float
//     [NemotronBlock.mlp] with quantized matmuls.
//   - PARTIAL split-half RoPE: only the first rotaryDim channels of each head rotate
//     ([partialRoPE], RotaryDim from rope.dimension_count); the tail passes through.
//
// The residual is SEQUENTIAL (Llama-style two-norm block), attention is bias-free
// causal GQA at the standard 1/√head_dim scale, and the LM head is UNTIED
// (output.weight, required — the nemotron architecture has no tied-embedding
// fallback).
//
// Dtype discipline (the §B quant-bias lesson): the quantized residual stream is f32 —
// QuantLinear outputs f32 activations — so every small tensor that touches it is f32
// too: the LayerNorm γ/β pairs and the dequantized embedding table. Real
// llama.cpp-quantized nemotron files store all of these as F32 (1-D tensors are never
// block-quantized), so the GGUF path and [QuantizeNemotron] land on identical bytes.
// Inference-only: the quantized weights are frozen and bypass the tape.
//
// Load a llama.cpp-quantized checkpoint with [QuantNemotronFromGGUF], or quantize a
// float model with [QuantizeNemotron].
type QuantNemotron struct {
	Config NemotronConfig        // model geometry, shared with float Nemotron (see NemotronConfig)
	TokEmb *tensor.Tensor        // [vocab, dim] f32 token embedding (lookup only)
	Blocks []*QuantNemotronBlock // the quantized Nemotron blocks
	Norm   *nn.LayerNorm         // final LayerNorm1P (f32 γ=1+w folded, β)
	Out    *nn.QuantLinear       // untied LM head (output.weight, required)
}

// QuantNemotronBlock is one sequential-residual QuantNemotron block: f32 LayerNorm1P
// pairs (γ pre-folded (1+w), bias β) gating quantized bias-free attention projections
// and a quantized bias-free ReLU² MLP (up/down only — the nemotron architecture has no
// gate projection). The QuantLinear weights carry the bulk of the bytes; the norm
// vectors stay f32, llama.cpp's on-disk convention for 1-D tensors.
type QuantNemotronBlock struct {
	InputNorm      *nn.LayerNorm   // input_layernorm (f32 γ=1+w, β), feeds attention
	PostAttnNorm   *nn.LayerNorm   // post_attention_layernorm (f32 γ=1+w, β), feeds the MLP
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (no bias)
	Wup, Wdown     *nn.QuantLinear // quantized ReLU² MLP up/down (no gate, no bias)
}

// QuantizeNemotron builds a QuantNemotron from a float Nemotron by quantizing every
// 2-D projection (attn q/k/v/o, mlp up/down, the untied head) to qt — the projections
// carry the bulk of the weights and compute, so this is where quantization pays off.
// Everything 1-D stays f32: the LayerNorm1P γ/β pairs (tiny, precision-sensitive, and
// — matching llama.cpp's on-disk convention — never block-quantized; the in-memory γ
// is ALREADY the folded (1 + hf_weight) gain, cloned as-is), plus the token embedding
// table (its lookup needs a float table; Nemotron's head is untied, so unlike
// [QuantizeGemma] the table itself need not be quantized). This makes the result
// byte-comparable to [QuantNemotronFromGGUF] on a file that quantizes exactly the 2-D
// projections — the exact-anchor gate. Each projection's inner dimension must be a
// multiple of qt's block size (32 for Q8_0/Q4_0, 256 for the k-quants).
func QuantizeNemotron(m *Nemotron, qt gguf.QuantType) (*QuantNemotron, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantNemotron{
		Config: m.Config,
		TokEmb: f32Clone(m.TokEmb),
		Norm:   f32LayerNorm(m.Norm), // γ stays the folded (1+w) gain; β carried alongside
	}
	var err error
	for _, b := range m.Blocks {
		qb := &QuantNemotronBlock{
			InputNorm:    f32LayerNorm(b.InputNorm),
			PostAttnNorm: f32LayerNorm(b.PostAttnNorm),
		}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeNemotron Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeNemotron Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeNemotron Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeNemotron Wo: %w", err)
		}
		if qb.Wup, err = mkQ(b.Wup); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeNemotron up_proj: %w", err)
		}
		if qb.Wdown, err = mkQ(b.Wdown); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeNemotron down_proj: %w", err)
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Out, err = mkQ(m.Out); err != nil {
		return nil, fmt.Errorf("nlp: QuantizeNemotron output: %w", err)
	}
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It
// mirrors Nemotron.Forward exactly — embedding lookup (no positional embedding, no
// scale), then per block the sequential residual
//
//	x = x + attention(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x))
//
// with bias-free quantized projections, PARTIAL split-half RoPE (only the first
// rotaryDim channels of each head rotate) and causal GQA, then the final LayerNorm1P
// and the quantized untied head — but every projection is a quantized in-kernel
// matmul, all activations f32.
func (m *QuantNemotron) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: Nemotron prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
		if q, err = partialRoPE(ctx, q, cfg.Heads, rot, backend.RoPEAttrs{Base: cfg.RopeBase}); err != nil {
			return nil, err
		}
		if k, err = partialRoPE(ctx, k, kv, rot, backend.RoPEAttrs{Base: cfg.RopeBase}); err != nil {
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
		// MLP sublayer: post_attention_layernorm → ReLU² 2-layer → residual add.
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

// mlp runs the bias-free ReLU² feed-forward on the normalized input xn [seq, dim]:
// down(relu²(up(x))) with relu²(y) = relu(y)² and no gate — the quantized twin of
// [NemotronBlock.mlp], computed with the same kernel sequence (matmul → ReLU →
// elementwise square via OpMul → matmul).
func (m *QuantNemotron) mlp(ctx *backend.Context, b *QuantNemotronBlock, xn *tensor.Tensor) (*tensor.Tensor, error) {
	u, err := b.Wup.Forward(ctx, xn)
	if err != nil {
		return nil, err
	}
	r, err := exec1a(ctx, backend.OpReLU, nil, u)
	if err != nil {
		return nil, err
	}
	r2, err := exec2(ctx, backend.OpMul, nil, r, r)
	if err != nil {
		return nil, err
	}
	return b.Wdown.Forward(ctx, r2)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this
// QuantNemotron. Reuses [NemotronCache] — the cache structure is identical to the
// float model's (post-partial-RoPE keys, raw values); only the projections differ.
func (m *QuantNemotron) NewCache() *NemotronCache {
	return &NemotronCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token (Nemotron has no
// embedding scale).
func (m *QuantNemotron) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized Nemotron by one token at absolute position pos,
// appending its post-partial-RoPE K and raw V to the cache and returning the
// next-token logits [1, vocab]. It mirrors Nemotron.DecodeStep exactly — single-token
// embedding, bias-free quantized projections, partial RoPE at PosOffset=pos (the
// unrotated tail channels pass through untouched), the single query attending to the
// whole cache without a causal mask, the ReLU² MLP, final LayerNorm1P, untied head —
// so a KV-cache decode matches the full Forward (same kernel sequence per row).
// Inference-only, like the rest of the type.
func (m *QuantNemotron) DecodeStep(ctx *backend.Context, cache *NemotronCache, token, pos int) (*tensor.Tensor, error) {
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
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	attnA := backend.Attrs(backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false})
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
		if q, err = partialRoPE(ctx, q, cfg.Heads, rot, backend.RoPEAttrs{Base: cfg.RopeBase, PosOffset: pos}); err != nil {
			return nil, err
		}
		if k, err = partialRoPE(ctx, k, kv, rot, backend.RoPEAttrs{Base: cfg.RopeBase, PosOffset: pos}); err != nil {
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
func (m *QuantNemotron) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
func (m *QuantNemotron) Close() error {
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
