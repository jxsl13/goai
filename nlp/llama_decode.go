package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LlamaCache holds the per-layer key/value tensors accumulated during autoregressive
// Llama decoding, so each new token attends to the cached past instead of recomputing
// attention over the whole prefix. The cached keys already carry their RoPE rotation
// (applied at the position each token entered the cache). K[l] and V[l] are
// contiguous [t, width] views over internal amortized-growth row buffers
// (rowBuf), refreshed each DecodeStep — appends cost O(1) per token instead of
// the old concatRows O(t) reallocate-and-copy.
type LlamaCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
	bufs kvBufs           // backing row buffers behind the K, V views
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *Llama) NewCache() *LlamaCache {
	return &LlamaCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *LlamaCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token].
func (m *Llama) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.TokEmb, token, d), nil
}

// DecodeStep advances the Llama by one token using the KV-cache and returns the
// next-token logits [1,vocab]. pos is the token's absolute position (== cache length
// before the call), used as the RoPE offset. The token's rotated k,v are appended to
// the cache and the single query attends to all cached keys. Inference-only (no tape);
// it produces the same logits as a full Forward over the prefix.
func (m *Llama) DecodeStep(ctx *backend.Context, cache *LlamaCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// Granite: inputs_embeds *= embedding_multiplier, matching hiddenFromEmbed; no-op for
	// Llama (EmbeddingMult 0). Without it a KV-cached Granite decode would diverge from Forward.
	if x, err = scaleScalar(ctx, x, cfg.EmbeddingMult); err != nil {
		return nil, err
	}
	// Granite: pre-softmax attention scale = attention_multiplier (√headDim cancels OpMHA's
	// built-in 1/√headDim), matching hiddenFromEmbed; default 1/√headDim when AttentionMult 0.
	attnScale := 0.0
	if cfg.AttentionMult != 0 {
		attnScale = cfg.AttentionMult * math.Sqrt(float64(cfg.Dim/cfg.Heads))
	}
	for l, b := range m.Blocks {
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
		// Qwen2-family q/k/v projection biases (added before RoPE, matching the full-sequence
		// path in hiddenFromEmbed); nil for Llama/Mistral. Without this the KV-cached decode
		// would silently drop the bias and diverge from Forward on a Qwen2 model.
		if q, err = addBiasIf(ctx, q, b.Bq); err != nil {
			return nil, err
		}
		if k, err = addBiasIf(ctx, k, b.Bk); err != nil {
			return nil, err
		}
		if v, err = addBiasIf(ctx, v, b.Bv); err != nil {
			return nil, err
		}
		// Qwen3 per-head QK-norm (before RoPE), matching hiddenFromEmbed; nil otherwise.
		if q, err = applyQKNorm(ctx, q, b.QNorm, cfg.Heads); err != nil {
			return nil, err
		}
		if k, err = applyQKNorm(ctx, k, b.KNorm, kv); err != nil {
			return nil, err
		}
		// RoPE the single token at its absolute position, then append k,v to the cache
		if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}, k); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query at the last position attends to all cached keys → no causal mask
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false, Scale: attnScale}, q, kNew, vNew)
		if err != nil {
			return nil, err
		}
		o, err := project(ctx, a, b.Wo)
		if err != nil {
			return nil, err
		}
		// Granite: x = x + o·residual_multiplier, matching hiddenFromEmbed (no-op for Llama).
		if o, err = scaleScalar(ctx, o, cfg.ResidualMult); err != nil {
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
		// Granite: x = x + ff·residual_multiplier (same scalar as the attention residual).
		if ff, err = scaleScalar(ctx, ff, cfg.ResidualMult); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.Out)
	if err != nil {
		return nil, err
	}
	// Granite: logits /= logits_scaling, matching ForwardFromEmbed (no-op for Llama).
	return divLogits(ctx, logits, cfg.LogitsScale)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler
// s, using the KV-cache (one forward per new token). Returns prompt+generated. With a
// greedy sampler the output is identical to argmax-ing a full Forward at each step.
// The decode runs on backend.Default() unless WithBackend overrides it (§T361).
func (m *Llama) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
