package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantOLMo2 is an [OLMo2] whose projection weights stay QUANTIZED (nn.QuantLinear)
// and are never materialized as full-precision matrices — the memory-efficient form of
// a llama.cpp-quantized OLMo 2 checkpoint, the olmo2 sibling of [QuantLlama] /
// [QuantStarCoder2]. Its forward mirrors OLMo2.Forward faithfully, both OLMo 2
// departures included:
//
//   - POST-norm blocks: the sublayers read the RAW residual stream (there is NO
//     input_layernorm) and their OUTPUT is normed BEFORE the residual add. Per block:
//
//     residual = x; a = attention(x);  a = post_attention_layernorm(a);   x = residual + a
//     residual = x; f = FFN(x);        f = post_feedforward_layernorm(f); x = residual + f
//
//   - FULL-WIDTH QK-norm: an f32 RMSNorm is applied to the ENTIRE q_proj / k_proj
//     output (over heads·head_dim and kv_heads·head_dim respectively) BEFORE the head
//     split and RoPE — q = q_norm(Wq·x), k = k_norm(Wk·x). Because the norm spans the
//     full projection width (not per-head like Qwen3), it is a plain
//     [nn.RMSNorm.Forward] on the [seq, heads·head_dim] tensor, exactly the float
//     attention's order.
//
// The RMSNorms are standard (γ·x̂, NOT Gemma's (1+γ)), everything is bias-free,
// attention is causal GQA at the standard 1/√head_dim scale with full NEOX split-half
// RoPE, the FFN is a bias-free SwiGLU ([nn.QuantSwiGLU]) and the LM head is UNTIED
// (output.weight — the OLMo 2 default; the GGUF arch materializes a tied 1B's head).
//
// Dtype discipline (the §B quant-bias lesson): the quantized residual stream is f32 —
// QuantLinear outputs f32 activations — so every small tensor that touches it is f32
// too: all five RMSNorm gains per block region (q/k norms, the two post-norms, the
// final norm) and the dequantized embedding table. Real llama.cpp-quantized olmo2
// files store all of these as F32 (1-D tensors are never block-quantized), so the GGUF
// path and [QuantizeOLMo2] land on identical bytes. Inference-only: the quantized
// weights are frozen and bypass the tape.
//
// Load a llama.cpp-quantized checkpoint with [QuantOLMo2FromGGUF], or quantize a float
// model with [QuantizeOLMo2].
type QuantOLMo2 struct {
	Config OLMo2Config
	TokEmb *tensor.Tensor // [vocab, dim] f32 token embedding (lookup only)
	Blocks []*QuantOLMo2Block
	Norm   *nn.RMSNorm     // final pre-logits RMSNorm (f32 gain)
	Out    *nn.QuantLinear // untied LM head (output.weight, required)
}

// QuantOLMo2Block is one post-norm QuantOLMo2 block: quantized bias-free attention
// projections and a quantized bias-free SwiGLU FFN reading the RAW residual, with f32
// RMSNorms on the sublayer OUTPUTS (PostAttnNorm/PostFFNNorm, applied pre-residual)
// and f32 full-width QNorm/KNorm on the whole q/k projections (applied before the head
// split and RoPE). The QuantLinear weights carry the bulk of the bytes; the norm gains
// stay f32, llama.cpp's on-disk convention for 1-D tensors.
type QuantOLMo2Block struct {
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (no bias)
	QNorm, KNorm   *nn.RMSNorm     // full-width q/k RMSNorm (f32, over heads·headDim / kv·headDim) before RoPE
	PostAttnNorm   *nn.RMSNorm     // post_attention_layernorm (f32, on the attention output, pre-residual)
	FFN            *nn.QuantSwiGLU // quantized SwiGLU feed-forward (no bias)
	PostFFNNorm    *nn.RMSNorm     // post_feedforward_layernorm (f32, on the FFN output, pre-residual)
}

// QuantizeOLMo2 builds a QuantOLMo2 from a float OLMo2 by quantizing every 2-D
// projection (attn q/k/v/o, the SwiGLU gate/up/down, the untied head) to qt — the
// projections carry the bulk of the weights and compute, so this is where quantization
// pays off. Everything 1-D stays f32: the RMSNorm gains (q/k norms, post-norms, final
// norm — tiny, precision-sensitive, and, matching llama.cpp's on-disk convention,
// never block-quantized), plus the token embedding table (its lookup needs a float
// table; OLMo 2's head is untied, so unlike [QuantizeGemma] the table itself need not
// be quantized). This makes the result byte-comparable to [QuantOLMo2FromGGUF] on a
// file that quantizes exactly the 2-D projections — the exact-anchor gate. The
// decoupled-HeadDim float form (config.head_dim ≠ Dim/Heads) quantizes fine — the
// forward is projection-width-driven — though the GGUF olmo2 convention itself is
// coupled (llama.cpp asserts n_rot == n_embd/n_head). Each projection's inner
// dimension must be a multiple of qt's block size (32 for Q8_0/Q4_0, 256 for the
// k-quants).
func QuantizeOLMo2(m *OLMo2, qt gguf.QuantType) (*QuantOLMo2, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantOLMo2{
		Config: m.Config,
		TokEmb: f32Clone(m.TokEmb),
		Norm:   f32RMSNorm(m.Norm),
	}
	var err error
	for _, b := range m.Blocks {
		qb := &QuantOLMo2Block{
			QNorm:        f32RMSNorm(b.QNorm),
			KNorm:        f32RMSNorm(b.KNorm),
			PostAttnNorm: f32RMSNorm(b.PostAttnNorm),
			PostFFNNorm:  f32RMSNorm(b.PostFFNNorm),
		}
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeOLMo2 Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeOLMo2 Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeOLMo2 Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeOLMo2 Wo: %w", err)
		}
		gate, err := mkQ(b.FFN.Wgate)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeOLMo2 ffn_gate: %w", err)
		}
		up, err := mkQ(b.FFN.Wup)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeOLMo2 ffn_up: %w", err)
		}
		down, err := mkQ(b.FFN.Wdown)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeOLMo2 ffn_down: %w", err)
		}
		qb.FFN = &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Out, err = mkQ(m.Out); err != nil {
		return nil, fmt.Errorf("nlp: QuantizeOLMo2 output: %w", err)
	}
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It
// mirrors OLMo2.Forward exactly — embedding lookup (no positional embedding, no
// scale), then per block the POST-norm residual
//
//	x = x + post_attention_layernorm(attention(x)); x = x + post_feedforward_layernorm(FFN(x))
//
// where attention reads the RAW residual and applies the full-width QK-norms before
// RoPE and causal GQA, then the final RMSNorm and the quantized untied head — but
// every projection is a quantized in-kernel matmul, all activations f32.
func (m *QuantOLMo2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: OLMo2 prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
		// Attention sublayer (post-norm): a = post_attention_layernorm(attn(x)); x = x + a.
		// Attention reads the RAW residual x — there is no input_layernorm.
		a, err := m.attention(ctx, b, x, attn)
		if err != nil {
			return nil, err
		}
		if a, err = b.PostAttnNorm.Forward(ctx, a); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// FFN sublayer (post-norm): f = post_feedforward_layernorm(SwiGLU(x)); x = x + f.
		f, err := b.FFN.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if f, err = b.PostFFNNorm.Forward(ctx, f); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// attention computes the quantized OLMo 2 attention over the raw residual x [seq, dim]:
// quantized q/k/v projections, the full-width f32 q_norm/k_norm over the WHOLE q/k
// projections (before the head split), full NEOX RoPE, causal GQA via OpMHA and the
// quantized o_proj — the quantized twin of [OLMo2.attention], same kernel order.
func (m *QuantOLMo2) attention(ctx *backend.Context, b *QuantOLMo2Block, x *tensor.Tensor, attn backend.AttnAttrs) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	q, err := b.Wq.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	k, err := b.Wk.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	v, err := b.Wv.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	// Full-width QK-norm: RMSNorm over the ENTIRE q/k projection (last axis =
	// heads·headDim for q, kv·headDim for k), applied BEFORE the head split and RoPE.
	if q, err = b.QNorm.Forward(ctx, q); err != nil {
		return nil, err
	}
	if k, err = b.KNorm.Forward(ctx, k); err != nil {
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
	return b.Wo.Forward(ctx, a)
}

// NewCache allocates an empty KV-cache for autoregressive decoding of this QuantOLMo2.
// Reuses [OLMo2Cache] — the cache structure is identical to the float model's
// (full-width-k_norm'd, post-RoPE keys, raw values); only the projections differ.
func (m *QuantOLMo2) NewCache() *OLMo2Cache {
	return &OLMo2Cache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token (OLMo 2 has no
// embedding scale).
func (m *QuantOLMo2) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized OLMo 2 by one token at absolute position pos,
// appending its full-width-k_norm'd, post-RoPE K and raw V to the cache and returning
// the next-token logits [1, vocab]. It mirrors OLMo2.DecodeStep exactly — single-token
// embedding, attention on the RAW residual with the full-width QK-norms before RoPE at
// PosOffset=pos, the single query attending to the whole cache without a causal mask,
// the attention output post-normed BEFORE the residual add (and the FFN output
// likewise), final RMSNorm, untied head — so a KV-cache decode matches the full
// Forward (same kernel sequence per row). Inference-only, like the rest of the type.
func (m *QuantOLMo2) DecodeStep(ctx *backend.Context, cache *OLMo2Cache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		// Attention sublayer (post-norm) on the RAW residual x.
		q, err := b.Wq.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		k, err := b.Wk.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		v, err := b.Wv.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		// Full-width QK-norm over the whole q/k projection (before the head split and
		// RoPE), matching the full-sequence path in Forward.
		if q, err = b.QNorm.Forward(ctx, q); err != nil {
			return nil, err
		}
		if k, err = b.KNorm.Forward(ctx, k); err != nil {
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
		// single query attends to all cached keys → no causal mask
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false}, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = b.Wo.Forward(ctx, a); err != nil {
			return nil, err
		}
		if a, err = b.PostAttnNorm.Forward(ctx, a); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// FFN sublayer (post-norm) on the RAW residual x.
		f, err := b.FFN.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if f, err = b.PostFFNNorm.Forward(ctx, f); err != nil {
			return nil, err
		}
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
func (m *QuantOLMo2) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
func (m *QuantOLMo2) Close() error {
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
