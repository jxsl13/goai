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
		// FFN sublayer (GeGLU)
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
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Tied LM head: logits = hidden · embedᵀ (unscaled table).
	return exec1(ctx, backend.OpMatMul, nil, x, m.tiedHead())
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler
// s, using the KV-cache (one forward per new token). Returns prompt+generated. With a
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
