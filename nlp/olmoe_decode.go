package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// OLMoECache holds the per-layer key/value tensors accumulated during autoregressive
// OLMoE decoding, so each new token attends to the cached past instead of recomputing
// attention over the whole prefix. The cached keys already carry both the full-width
// k_norm and their RoPE rotation (applied at the position each token entered the cache).
// The cache concerns only the attention sublayer; the sparse-MoE FFN is stateless and
// re-routes each token afresh.
type OLMoECache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
	bufs kvBufs           // backing row buffers behind the K, V views (amortized-O(1) append)
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *OLMoE) NewCache() *OLMoECache {
	return &OLMoECache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *OLMoECache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token] (OLMoE has
// no embedding scale).
func (m *OLMoE) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.TokEmb, token, d), nil
}

// DecodeStep advances the OLMoE by one token using the KV-cache and returns the
// next-token logits [1,vocab]. pos is the token's absolute position (== cache length
// before the call), used as the RoPE offset. The token flows through the pre-norm blocks
// — x = x + attn(input_layernorm(x)), then x = x + moe(post_attention_layernorm(x)) — with
// the freshly full-width q/k-normed and rotated k,v appended to the cache and the single
// query attending to all cached keys (no causal mask needed). The FFN is the sparse top-k
// MoE via [nn.SparseMoE.ForwardDecode] (only the routed experts are evaluated), which
// selects the SAME experts prefill would, so routing is bit-identical. Inference-only (no
// tape); it produces the same logits as a full Forward over the prefix.
func (m *OLMoE) DecodeStep(ctx *backend.Context, cache *OLMoECache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// Step-invariant attrs boxed once here instead of ~N_layers times inside the loop
	// (T956, the T955/T957 per-layer-Attrs-boxing hoist applied to OLMoE decode).
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos})
	attn := backend.Attrs(backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false})
	for l, b := range m.Blocks {
		// Attention sublayer (pre-norm) on the input_layernorm'd residual.
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := project(ctx, xb, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := project(ctx, xb, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := project(ctx, xb, b.Wv)
		if err != nil {
			return nil, err
		}
		// Full-width QK-norm over the whole q/k projection (before the head split and RoPE),
		// matching the full-sequence path in hidden. Without this the KV-cached decode would
		// drop the norm and diverge from Forward.
		if q, err = b.QNorm.Forward(ctx, q); err != nil {
			return nil, err
		}
		if k, err = b.KNorm.Forward(ctx, k); err != nil {
			return nil, err
		}
		// RoPE the single token at its absolute position, then append k,v to the cache.
		if q, err = exec1(ctx, backend.OpRoPE, qRoPE, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query attends to all cached keys → no causal mask
		a, err := exec1(ctx, backend.OpMHA, attn, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		o, err := project(ctx, a, b.Wo)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// sparse-MoE FFN sublayer (pre-norm); gateLogits ignored during inference.
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		// Sparse decode: evaluate only the top-k routed experts, not all E (identical output).
		ff, _, err := b.MoE.ForwardDecode(ctx, xf)
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
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// Prefill runs the transformer ONCE over the whole prompt, seeds the KV-cache with
// every prompt token's key/value rows, and returns the full logits [seq, vocab]
// (row seq−1 is the next-token distribution Generate samples from). It is the
// batched replacement for feeding the prompt token-by-token through
// [OLMoE.DecodeStep]: one full-sequence pass through the block stack instead of
// seq single-token dispatch rounds, with a bit-identical cache — the captured k/v
// are the same full-width-k_norm'd, POST-RoPE rows DecodeStep would have appended
// (full-sequence RoPE rotates row p for position p, exactly DecodeStep's
// PosOffset=p). The FFN runs the SAME sparse [nn.SparseMoE.ForwardDecode] path as
// DecodeStep, batched over all prompt rows — one routing pass and at most one
// dispatch per routed expert replaces seq per-token sparse-routing rounds, and
// using the identical kernel sequence is what keeps the cache bit-identical (the
// dense combine kernel's fused multiply-add rounds ~1 ulp differently, §B64).
//
// For readers new to LLM inference: generation has two phases. "Prefill" processes
// the whole prompt at once — every token is known up front, so the model can attend
// over the full sequence in a handful of large batched kernels and store each
// layer's keys/values in the cache. "Decode" then produces one token at a time,
// where each new token only needs its own small computation plus the cached past.
// Batching the prefill is the standard first optimization of every serving stack.
//
// cache must be empty (a fresh [OLMoE.NewCache]); Prefill errors on a non-empty
// cache because the full-sequence RoPE and causal mask assume positions 0..seq−1.
// Inference-only, like DecodeStep: run it on a plain backend.NewContext.
func (m *OLMoE) Prefill(ctx *backend.Context, cache *OLMoECache, tokens []int) (*tensor.Tensor, error) {
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
	}, true) // sparseFFN: DecodeStep's exact MoE kernel sequence (bit-parity, §B64)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, h, m.Out)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler s,
// using the KV-cache (one batched [OLMoE.Prefill] over the prompt, then one
// [OLMoE.DecodeStep] per new token). Returns prompt+generated. With a greedy
// sampler the output is identical to argmax-ing a full Forward at each step. The decode
// runs on backend.Default() unless WithBackend overrides it.
func (m *OLMoE) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
