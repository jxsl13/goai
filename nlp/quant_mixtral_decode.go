package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// NewCache allocates an empty KV-cache for autoregressive decoding of this QuantMixtral
// (one post-RoPE K/V slot per block, filled as tokens are decoded). Reuses
// [MixtralCache] — the cache structure is identical to the float model's; only the
// projections differ. The sparse-MoE FFN is stateless (each token is routed afresh), so
// the cache concerns only the attention sublayer, exactly as in the float pipeline.
func (m *QuantMixtral) NewCache() *MixtralCache {
	return &MixtralCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token.
func (m *QuantMixtral) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	return embedRow(m.TokEmb, token, m.Config.Dim), nil
}

// DecodeStep advances the quantized Mixtral by one token at absolute position pos,
// appending its post-RoPE K/V to the cache and returning the next-token logits
// [1, vocab]. The attention mirrors [QuantLlama.DecodeStep] — quantized projections,
// RoPE at PosOffset=pos, the single query attending to the whole cache — and the FFN is
// the SPARSE top-k MoE ([QuantMoE.Forward] on one row): the router scores all E
// experts, but only the k it selects are evaluated, skipping (E−k)/E of the expert
// GEMVs — the dominant per-token cost of an MoE decode (4× fewer FFN weights touched
// for Mixtral's k=2 of E=8). Because Forward runs the identical sparse kernel sequence,
// a KV-cache decode matches the full Forward up to f32 reassociation. Inference-only.
func (m *QuantMixtral) DecodeStep(ctx *backend.Context, cache *MixtralCache, token, pos int) (*tensor.Tensor, error) {
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
		// Optional per-head QK-norm before RoPE, matching Forward; nil is a no-op.
		if q, err = applyQKNorm(ctx, q, b.QNorm, cfg.Heads); err != nil {
			return nil, err
		}
		if k, err = applyQKNorm(ctx, k, b.KNorm, kv); err != nil {
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
		// single query at the last position attends to all cached keys → no causal mask
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
		// sparse-MoE FFN: only the top-k routed experts run for this token
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.MoE.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on the
// quantized model, using the KV-cache (each step is one token with sparse top-k expert
// evaluation, not a full re-forward), and returns prompt+new. The sampler s selects
// each token (Greedy() for deterministic argmax). Stops at the context limit. The same
// shape as [QuantLlama.Generate].
func (m *QuantMixtral) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
