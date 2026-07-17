package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MPTCache holds the per-layer key/value tensors accumulated during autoregressive MPT
// decoding, so each new token attends to the cached past instead of recomputing attention
// over the whole prefix. MPT is an ALiBi model — there is no RoPE, so the cached keys carry
// no positional rotation; position enters solely through the per-head linear-distance bias
// [backend.OpMHA] adds at attention time. The cache concerns only the attention sublayer;
// the GELU MLP is stateless.
type MPTCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
	bufs kvBufs           // backing row buffers behind the K, V views (amortized-O(1) append)
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *MPT) NewCache() *MPTCache {
	return &MPTCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *MPTCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token]. MPT applies no
// embedding scale and has no positional embedding — position enters through ALiBi.
func (m *MPT) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.TokEmb, token, d), nil
}

// DecodeStep advances the MPT by one token using the KV-cache and returns the next-token
// logits [1,vocab]. pos is the token's absolute position (== cache length before the call).
// MPT is an ALiBi model, so pos is NOT used to rotate q/k (there is no RoPE); instead the
// single query at absolute position pos attends the cached keys 0..pos, and [backend.OpMHA]'s
// ALiBi bias slopeₕ·(j−pos) is recovered automatically from the query/key length gap
// (off = sk−sq = pos, so the single query row's absolute position is pos). The token flows
// through the SEQUENTIAL-residual blocks — x = x + attn(norm_1(x)); x = x + ffn(norm_2(x)) —
// with its k,v appended to the cache. Inference-only (no tape); it produces the same logits
// as a full Forward over the prefix.
func (m *MPT) DecodeStep(ctx *backend.Context, cache *MPTCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// Single query vs all cached keys: no causal mask (every cached key is at pos ≤ this
	// token). ALiBi is still on — the bias depends on the query/key distance, which OpMHA
	// derives from the sq/sk gap, so the single query is correctly biased at absolute pos.
	attn := backend.AttnAttrs{Heads: cfg.Heads, Causal: false, ALiBi: true}
	for l, b := range m.Blocks {
		// Attention sublayer over norm_1(x).
		an, err := b.Norm1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := project(ctx, an, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := project(ctx, an, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := project(ctx, an, b.Wv)
		if err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		a, err := exec1(ctx, backend.OpMHA, attn, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = project(ctx, a, b.Wo); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// MLP sublayer over norm_2(x).
		fn, err := b.Norm2.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, fn)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Tied LM head: logits = hidden · wteᵀ (Out stored [dim, vocab]).
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler s,
// using the KV-cache (one forward per new token). Returns prompt+generated. With a greedy
// sampler the output is identical to argmax-ing a full Forward at each step. The decode runs
// on backend.Default() unless WithBackend overrides it.
func (m *MPT) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
