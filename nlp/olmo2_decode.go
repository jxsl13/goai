package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// OLMo2Cache holds the per-layer key/value tensors accumulated during autoregressive
// OLMo 2 decoding, so each new token attends to the cached past instead of recomputing
// attention over the whole prefix. The cached keys already carry both the full-width
// k_norm and their RoPE rotation (applied at the position each token entered the cache).
type OLMo2Cache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
	bufs kvBufs           // backing row buffers behind the K, V views (amortized-O(1) append)
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *OLMo2) NewCache() *OLMo2Cache {
	return &OLMo2Cache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *OLMo2Cache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token] (OLMo 2 has
// no embedding scale).
func (m *OLMo2) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.TokEmb, token, d), nil
}

// DecodeStep advances the OLMo 2 by one token using the KV-cache and returns the
// next-token logits [1,vocab]. pos is the token's absolute position (== cache length
// before the call), used as the RoPE offset. The token flows through the post-norm blocks
// — attention(x) → post_attention_layernorm → +residual, then SwiGLU(x) →
// post_feedforward_layernorm → +residual — with the freshly q/k-normed and rotated k,v
// appended to the cache and the single query attending to all cached keys (no causal mask
// needed). Inference-only (no tape); it produces the same logits as a full Forward.
func (m *OLMo2) DecodeStep(ctx *backend.Context, cache *OLMo2Cache, token, pos int) (*tensor.Tensor, error) {
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
		q, err := project(ctx, x, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := project(ctx, x, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := project(ctx, x, b.Wv)
		if err != nil {
			return nil, err
		}
		// Full-width QK-norm over the whole q/k projection (before the head split and RoPE),
		// matching the full-sequence path in hidden.
		if q, err = b.QNorm.Forward(ctx, q); err != nil {
			return nil, err
		}
		if k, err = b.KNorm.Forward(ctx, k); err != nil {
			return nil, err
		}
		// RoPE the single token at its absolute position, then append k,v to the cache.
		//perfscan:ignore PS6016,PS6017 RoPE attrs-literal dispatch, kernel-dominated | RoPE op dispatch, kernel dominates, no lever
		if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016,PS6017 RoPE attrs-literal dispatch, kernel-dominated | RoPE op dispatch, kernel dominates, no lever
		if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}, k); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query attends to all cached keys → no causal mask
		//perfscan:ignore PS6016,PS6017 MHA attrs-literal dispatch, kernel-dominated | MHA op dispatch, kernel dominates, no lever
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false}, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = project(ctx, a, b.Wo); err != nil {
			return nil, err
		}
		if a, err = b.PostAttnNorm.Forward(ctx, a); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
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
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// Prefill runs the transformer ONCE over the whole prompt, seeds the KV-cache with
// every prompt token's key/value rows, and returns the full logits [seq, vocab]
// (row seq−1 is the next-token distribution Generate samples from). It is the
// batched replacement for feeding the prompt token-by-token through
// [OLMo2.DecodeStep]: one full-sequence pass through the post-norm block stack
// instead of seq single-token dispatch rounds, with a bit-identical cache — the
// captured k/v are the same full-width-k_norm'd, POST-RoPE rows DecodeStep would
// have appended (full-sequence RoPE rotates row p for position p, exactly
// DecodeStep's PosOffset=p), so decoding may continue from the seeded cache as if
// the prompt had been stepped through one token at a time.
//
// For readers new to LLM inference: generation has two phases. "Prefill" processes
// the whole prompt at once — every token is known up front, so the model can attend
// over the full sequence in a handful of large batched kernels and store each
// layer's keys/values in the cache. "Decode" then produces one token at a time,
// where each new token only needs its own small computation plus the cached past.
// Batching the prefill is the standard first optimization of every serving stack.
//
// cache must be empty (a fresh [OLMo2.NewCache]); Prefill errors on a non-empty
// cache because the full-sequence RoPE and causal mask assume positions 0..seq−1.
// Inference-only, like DecodeStep: run it on a plain backend.NewContext.
func (m *OLMo2) Prefill(ctx *backend.Context, cache *OLMo2Cache, tokens []int) (*tensor.Tensor, error) {
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
	return exec1(ctx, backend.OpMatMul, nil, h, m.Out)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler s,
// using the KV-cache (one batched [OLMo2.Prefill] over the prompt, then one
// [OLMo2.DecodeStep] per new token). Returns prompt+generated. With a greedy
// sampler the output is identical to argmax-ing a full Forward at each step. The decode
// runs on backend.Default() unless WithBackend overrides it.
func (m *OLMo2) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
