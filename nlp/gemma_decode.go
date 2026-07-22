package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GemmaCache holds the per-layer key/value tensors accumulated during autoregressive
// Gemma (v1) decoding, so each new token attends to the cached past instead of
// recomputing attention over the whole prefix. The cached keys already carry their
// RoPE rotation (applied at the position each token entered the cache). The cache
// concerns only the attention sublayer; the GeGLU FFN is stateless.
type GemmaCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
	bufs kvBufs           // backing row buffers behind the K, V views (amortized-O(1) append)
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *Gemma) NewCache() *GemmaCache {
	return &GemmaCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *GemmaCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the √dim-scaled embedding of a single token: x[1,dim] =
// √dim · TokEmb[token]. Gemma scales the embeddings by √dim (the "normalizer") right
// after lookup, on the residual stream only — the tied LM head still uses the UNSCALED
// table — so DecodeStep applies the scalar here exactly as [Gemma.Forward] does.
func (m *Gemma) embedOne(ctx *backend.Context, token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	x := embedRow(m.TokEmb, token, d)
	scale := tensor.New(tensor.F64, tensor.Shape{})
	scale.Storage().F64()[0] = math.Sqrt(float64(d))
	return exec1(ctx, backend.OpMul, nil, x, scale)
}

// DecodeStep advances the Gemma by one token using the KV-cache and returns the
// next-token logits [1,vocab]. pos is the token's absolute position (== cache length
// before the call), used as the RoPE offset. The token's √dim-scaled embedding flows
// through the pre-norm blocks: (1+w) RMSNorm is already folded into the loaded gains,
// the projections are bias-free, the freshly rotated k,v are appended to the cache and
// the single query attends to all cached keys (no causal mask), and the FFN is GeGLU.
// The tied LM head uses the UNSCALED embedding: logits = hidden · embedᵀ. Inference-only
// (no tape); it produces the same logits as a full Forward over the prefix.
func (m *Gemma) DecodeStep(ctx *backend.Context, cache *GemmaCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	x, err := m.embedOne(ctx, token)
	if err != nil {
		return nil, err
	}
	// These attrs are step-invariant (Base/Heads/KV/pos are the same for every layer), so
	// box each into the Attrs interface ONCE here instead of ~N_layers times inside the
	// loop — the T955 per-layer-Attrs-boxing hoist applied to Gemma decode (T956).
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos})
	attn := backend.Attrs(backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false})
	for l, b := range m.Blocks {
		// attention sublayer (bias-free)
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
		// RoPE the single token at its absolute position, then append k,v to the cache.
		if q, err = exec1(ctx, backend.OpRoPE, qRoPE, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query at the last position attends to all cached keys → no causal mask
		a, err := exec1(ctx, backend.OpMHA, attn, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		o, err := project(ctx, a, b.Wo)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// FFN sublayer (GeGLU)
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Tied LM head: logits = hidden · embedᵀ (unscaled table).
	return exec1(ctx, backend.OpMatMul, nil, x, m.tiedHead())
}

// Prefill runs the transformer ONCE over the whole prompt, seeds the KV-cache with
// every prompt token's key/value rows, and returns the full logits [seq, vocab]
// (row seq−1 is the next-token distribution Generate samples from). It is the
// batched replacement for feeding the prompt token-by-token through
// [Gemma.DecodeStep]: one full-sequence pass through the block stack instead of
// seq single-token dispatch rounds, with a bit-identical cache — the captured k/v
// are the same post-RoPE rows DecodeStep would have appended (full-sequence RoPE
// rotates row p for position p, exactly DecodeStep's PosOffset=p), the √dim
// embedding scale is applied identically, and the tied LM head uses the same
// cached UNSCALED-table transpose (tiedHead).
//
// For readers new to LLM inference: generation has two phases. "Prefill" processes
// the whole prompt at once — every token is known up front, so the model can attend
// over the full sequence in a handful of large batched kernels and store each
// layer's keys/values in the cache. "Decode" then produces one token at a time,
// where each new token only needs its own small computation plus the cached past.
// Batching the prefill is the standard first optimization of every serving stack.
//
// cache must be empty (a fresh [Gemma.NewCache]); Prefill errors on a non-empty
// cache because the full-sequence RoPE and causal mask assume positions 0..seq−1.
// Inference-only, like DecodeStep: run it on a plain backend.NewContext.
func (m *Gemma) Prefill(ctx *backend.Context, cache *GemmaCache, tokens []int) (*tensor.Tensor, error) {
	if n := cache.Len(); n != 0 {
		return nil, fmt.Errorf("nlp: Prefill needs an empty cache, got %d cached tokens", n)
	}
	if len(cache.K) < len(m.Blocks) {
		cache.K = growSlice(cache.K, len(m.Blocks))
	}
	if len(cache.V) < len(m.Blocks) {
		cache.V = growSlice(cache.V, len(m.Blocks))
	}
	return m.forwardCapture(ctx, tokens, func(l int, k, v *tensor.Tensor) {
		// One multi-row Append per layer: the whole prompt's [seq, kvWidth] k/v
		// land in the cache's row buffers in a single typed block copy.
		cache.K[l], cache.V[l] = cache.bufs.appendKV(cache.K, cache.V, l, k, v)
	})
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler
// s, using the KV-cache (one batched [Gemma.Prefill] over the prompt, then one
// [Gemma.DecodeStep] per new token). Returns prompt+generated. With a
// greedy sampler the output is identical to argmax-ing a full Forward at each step.
// The decode runs on backend.Default() unless WithBackend overrides it.
func (m *Gemma) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
