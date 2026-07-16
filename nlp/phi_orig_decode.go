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
	x := tensor.New(m.TokEmb.Dtype(), tensor.Shape{1, d})
	for j := range d {
		x.SetF64(m.TokEmb.AtF64(token, j), 0, j)
	}
	return x, nil
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
		kNew := concatRows(cache.K[l], k)
		vNew := concatRows(cache.V[l], v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query attends to all cached keys → no causal mask
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
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.logits(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler s, using
// the KV-cache (one forward per new token). Returns prompt+generated. With a greedy sampler the
// output is identical to argmax-ing a full Forward at each step. The decode runs on
// backend.Default() unless WithBackend overrides it.
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
