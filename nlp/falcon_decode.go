package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// FalconCache holds the per-layer key/value tensors accumulated during autoregressive Falcon
// decoding, so each new token attends to the cached past instead of recomputing attention over
// the whole prefix. The cached keys already carry their RoPE rotation (applied at the position
// each token entered the cache). Falcon is multi-query (MQA): each block caches a SINGLE key
// head and a single value head. The cache concerns only the attention sublayer; the GELU MLP
// is stateless.
type FalconCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *Falcon) NewCache() *FalconCache {
	return &FalconCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *FalconCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token]. Falcon applies
// no embedding scale and has no positional embedding — position enters through RoPE.
func (m *Falcon) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.TokEmb, token, d), nil
}

// DecodeStep advances the Falcon by one token using the KV-cache and returns the next-token
// logits [1,vocab]. pos is the token's absolute position (== cache length before the call),
// used as the RoPE offset. The token flows through the SINGLE-NORM PARALLEL-residual blocks —
// xn = input_layernorm(x); x = x + attention(xn) + mlp(xn) — with the freshly rotated single
// key/value head (MQA) appended to the cache and the single query attending all cached keys
// (no causal mask needed). Inference-only (no tape); it produces the same logits as a full
// Forward over the prefix.
func (m *Falcon) DecodeStep(ctx *backend.Context, cache *FalconCache, token, pos int) (*tensor.Tensor, error) {
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
		// Single-norm parallel residual: ONE norm feeds both sublayers.
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := project(ctx, xn, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := project(ctx, xn, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := project(ctx, xn, b.Wv)
		if err != nil {
			return nil, err
		}
		// Standard split-half RoPE at the token's absolute position, then append k,v.
		if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.ropeBase(), Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.ropeBase(), Heads: kv, PosOffset: pos}, k); err != nil {
			return nil, err
		}
		kNew := concatRows(cache.K[l], k)
		vNew := concatRows(cache.V[l], v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query attends all cached keys → no causal mask; MQA via KVHeads=1
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false}, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		if a, err = project(ctx, a, b.Wo); err != nil {
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
	// Untied LM head: logits = hidden · lm_head (Out stored [dim, vocab]).
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler s,
// using the KV-cache (one forward per new token). Returns prompt+generated. With a greedy
// sampler the output is identical to argmax-ing a full Forward at each step. The decode runs
// on backend.Default() unless WithBackend overrides it.
func (m *Falcon) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
