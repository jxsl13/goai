package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Qwen2MoeCache holds the per-layer key/value tensors accumulated during autoregressive
// Qwen2-MoE decoding, so each new token attends to the cached past instead of recomputing
// attention over the whole prefix. The cached keys already carry their RoPE rotation
// (applied at the position each token entered the cache). The cache concerns only the
// attention sublayer; the MoE-plus-shared-expert FFN is stateless and re-runs each token.
type Qwen2MoeCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *Qwen2MoE) NewCache() *Qwen2MoeCache {
	return &Qwen2MoeCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *Qwen2MoeCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token].
func (m *Qwen2MoE) embedOne(token int) (*tensor.Tensor, error) {
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

// DecodeStep advances the Qwen2-MoE by one token using the KV-cache and returns the
// next-token logits [1,vocab]. pos is the token's absolute position (== cache length before
// the call), the RoPE offset. The token's rotated k,v are appended to the cache and the single
// query attends to all cached keys; the FFN is the same sparse-MoE-plus-shared-expert block as
// [Qwen2MoE.Forward] (the router picks the same experts and the shared expert runs on the
// token), so this produces the same logits as a full Forward over the prefix. Inference-only.
func (m *Qwen2MoE) DecodeStep(ctx *backend.Context, cache *Qwen2MoeCache, token, pos int) (*tensor.Tensor, error) {
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
		// attention sublayer (Qwen2: biased q/k/v, bias-free o_proj)
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := projBias(ctx, xb, b.Wq, b.Bq)
		if err != nil {
			return nil, err
		}
		k, err := projBias(ctx, xb, b.Wk, b.Bk)
		if err != nil {
			return nil, err
		}
		v, err := projBias(ctx, xb, b.Wv, b.Bv)
		if err != nil {
			return nil, err
		}
		// RoPE the single token at its absolute position, then append k,v to the cache
		if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}, k); err != nil {
			return nil, err
		}
		kNew := concatRows(cache.K[l], k)
		vNew := concatRows(cache.V[l], v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query at the last position attends to all cached keys → no causal mask
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false}, q, kNew, vNew)
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
		// sparse-MoE + shared-expert FFN sublayer (identical to Forward)
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := m.ffn(ctx, b, xf)
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

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler s,
// using the KV-cache (one forward per new token). Returns prompt+generated. With a greedy
// sampler the output matches argmax-ing a full Forward at each step. The decode runs on
// backend.Default() unless WithBackend overrides it.
func (m *Qwen2MoE) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	if gc.be != nil {
		ctx = ctx.WithBackend(gc.be)
	}
	cache := m.NewCache()
	out := append([]int(nil), prompt...)
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}

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
