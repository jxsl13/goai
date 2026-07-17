package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// NewCache allocates an empty KV-cache for autoregressive decoding of this QuantLlama (one
// post-RoPE K/V slot per block, filled as tokens are decoded). Reuses LlamaCache — the cache
// structure is identical to the float model's; only the projections differ.
func (m *QuantLlama) NewCache() *LlamaCache {
	return &LlamaCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// embedOne returns the f32 embedding row [1, dim] of a single token.
func (m *QuantLlama) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	x := tensor.New(tensor.F32, tensor.Shape{1, d})
	for j := range d {
		x.SetF64(m.TokEmb.AtF64(token, j), 0, j)
	}
	return x, nil
}

// DecodeStep advances the quantized model by one token at absolute position pos, appending its
// post-RoPE K/V to the cache and returning the next-token logits [1, vocab]. It mirrors
// Llama.DecodeStep (§T121) exactly — quantized projections, RoPE at PosOffset=pos, single-query
// attention over the whole cache — so a KV-cache decode of a quantized model matches its full
// Forward up to f32 reassociation, all on the GPU.
func (m *QuantLlama) DecodeStep(ctx *backend.Context, cache *LlamaCache, token, pos int) (*tensor.Tensor, error) {
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
		// Qwen2 biases + Qwen3 QK-norm before RoPE, matching Forward (and Llama.DecodeStep);
		// nil fields are no-ops. Without this a KV-cached quantized decode of a Qwen model
		// would silently drop the extras and diverge from Forward.
		if q, k, v, err = applyQwenAttnExtras(ctx, b, q, k, v, cfg.Heads, kv); err != nil {
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
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on the quantized model,
// using the KV-cache (each step is one token, not a full re-forward), and returns prompt+new. The
// sampler s selects each token (Greedy() for deterministic argmax). Stops at the context limit.
func (m *QuantLlama) Generate(prompt []int, maxNew int, s TokenSampler) ([]int, error) {
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
		l, err := m.DecodeStep(ctx, cache, next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}
