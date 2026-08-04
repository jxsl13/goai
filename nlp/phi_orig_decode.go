package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// PhiCache holds the per-layer key/value tensors accumulated during autoregressive Phi
// decoding, so each new token attends to the cached past instead of recomputing attention
// over the whole prefix. The cached keys already carry their partial-RoPE rotation (applied at
// the position each token entered the cache).
type PhiCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
	bufs kvBufs           // backing row buffers behind the K, V views (amortized-O(1) append)
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *Phi) NewCache() *PhiCache {
	return &PhiCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *PhiCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token].
func (m *Phi) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.TokEmb, token, d), nil
}

// DecodeStep advances the Phi by one token using the KV-cache and returns the next-token
// logits [1,vocab]. pos is the token's absolute position (== cache length before the call),
// used as the partial-RoPE offset. The token flows through the single-norm parallel-residual
// blocks — xn = input_layernorm(x); x = attention(xn) + mlp(xn) + x — with the freshly rotated
// k,v appended to the cache and the single query attending to all cached keys (no causal mask
// needed). Inference-only (no tape); it produces the same logits as a full Forward.
func (m *Phi) DecodeStep(ctx *backend.Context, cache *PhiCache, token, pos int) (*tensor.Tensor, error) {
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
	for l, b := range m.Blocks {
		// One input_layernorm feeds both attention and MLP; both outputs plus the raw residual.
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := projBias(ctx, xn, b.Wq, b.Bq)
		if err != nil {
			return nil, err
		}
		k, err := projBias(ctx, xn, b.Wk, b.Bk)
		if err != nil {
			return nil, err
		}
		v, err := projBias(ctx, xn, b.Wv, b.Bv)
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
		//perfscan:ignore PS6016,PS6017 single OpMHA decode dispatch, compute inside kernel
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false}, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = projBias(ctx, a, b.Wdense, b.Bdense); err != nil {
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
	return m.logits(ctx, x)
}

// Prefill runs the transformer ONCE over the whole prompt, seeds the KV-cache with
// every prompt token's key/value rows, and returns the full logits [seq, vocab]
// (row seq−1 is the next-token distribution Generate samples from; every row already
// carries the lm_head bias). It is the batched replacement for feeding the prompt
// token-by-token through [Phi.DecodeStep]: one full-sequence pass through the
// single-norm parallel-residual block stack instead of seq single-token dispatch
// rounds, with a bit-identical cache — the captured k/v are the same post-bias,
// post-partial-RoPE rows DecodeStep would have appended (the full-sequence partial
// RoPE rotates row p for position p, exactly DecodeStep's PosOffset=p, and the
// unrotated tail channels pass through untouched on both paths), so decoding may
// continue from the seeded cache as if the prompt had been stepped through one
// token at a time.
//
// For readers new to LLM inference: generation has two phases. "Prefill" processes
// the whole prompt at once — every token is known up front, so the model can attend
// over the full sequence in a handful of large batched kernels and store each
// layer's keys/values in the cache. "Decode" then produces one token at a time,
// where each new token only needs its own small computation plus the cached past.
// Batching the prefill is the standard first optimization of every serving stack.
//
// cache must be empty (a fresh [Phi.NewCache]); Prefill errors on a non-empty
// cache because the full-sequence RoPE and causal mask assume positions 0..seq−1.
// Inference-only, like DecodeStep: run it on a plain backend.NewContext.
func (m *Phi) Prefill(ctx *backend.Context, cache *PhiCache, tokens []int) (*tensor.Tensor, error) {
	if n := cache.Len(); n != 0 {
		return nil, fmt.Errorf("nlp: Prefill needs an empty cache, got %d cached tokens", n)
	}
	if len(cache.K) < len(m.Blocks) {
		cache.K = growSlice(cache.K, len(m.Blocks))
	}
	if len(cache.V) < len(m.Blocks) {
		cache.V = growSlice(cache.V, len(m.Blocks))
	}
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	h, err := m.hiddenCapture(ctx, x, func(l int, k, v *tensor.Tensor) {
		// One multi-row Append per layer: the whole prompt's [seq, kvWidth] k/v
		// land in the cache's row buffers in a single typed block copy.
		cache.K[l], cache.V[l] = cache.bufs.appendKV(cache.K, cache.V, l, k, v)
	})
	if err != nil {
		return nil, err
	}
	// Untied, BIASED lm_head — the same logits helper DecodeStep and Forward use.
	return m.logits(ctx, h)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler s, using
// the KV-cache (one batched [Phi.Prefill] over the prompt, then one [Phi.DecodeStep] per new
// token). Returns prompt+generated. With a greedy sampler the output is identical to
// argmax-ing a full Forward at each step. The decode runs on backend.Default() unless
// WithBackend overrides it.
func (m *Phi) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	if gc.be != nil {
		ctx = ctx.WithBackend(gc.be)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}
	cache := m.NewCache()
	out := append([]int(nil), prompt...)

	// Batched prefill: one full-sequence pass seeds the cache and yields the
	// prompt's logits, replacing len(prompt) single-token DecodeStep rounds.
	full, err := m.Prefill(ctx, cache, prompt)
	if err != nil {
		return nil, err
	}
	logits, err := full.Slice(0, full.Shape()[0]-1, full.Shape()[0])
	if err != nil {
		return nil, err
	}
	pos := len(prompt)
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
