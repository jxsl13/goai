package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Gemma2Cache holds the per-layer key/value tensors accumulated during autoregressive
// Gemma 2 decoding, so each new token attends to the cached past instead of recomputing
// attention over the whole prefix. The cached keys already carry their RoPE rotation
// (applied at the position each token entered the cache). The cache concerns only the
// attention sublayer; the sandwich-normed GeGLU FFN is stateless.
type Gemma2Cache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *Gemma2) NewCache() *Gemma2Cache {
	return &Gemma2Cache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *Gemma2Cache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the √dim-scaled embedding of a single token: x[1,dim] =
// √dim · TokEmb[token]. Like Gemma, Gemma 2 scales the embeddings by √dim (the
// "normalizer") on the residual stream only — the tied LM head still uses the UNSCALED
// table — so DecodeStep applies the scalar here exactly as [Gemma2.Forward] does.
func (m *Gemma2) embedOne(ctx *backend.Context, token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	x := tensor.New(m.TokEmb.Dtype(), tensor.Shape{1, d})
	for j := range d {
		x.SetF64(m.TokEmb.AtF64(token, j), 0, j)
	}
	scale := tensor.New(tensor.F64, tensor.Shape{})
	scale.Storage().F64()[0] = math.Sqrt(float64(d))
	return exec1(ctx, backend.OpMul, nil, x, scale)
}

// DecodeStep advances the Gemma 2 by one token using the KV-cache and returns the
// next-token logits [1,vocab]. pos is the token's absolute position (== cache length
// before the call), used as the RoPE offset. The token flows through the sandwich-normed
// blocks — input_layernorm → capped attention → post_attention_layernorm → +residual,
// then pre_feedforward_layernorm → GeGLU → post_feedforward_layernorm → +residual — with
// the freshly rotated k,v appended to the cache and the single query attending to all
// cached keys via the soft-capped single-query attention (no causal mask needed). The
// final [1,vocab] logits are soft-capped. Inference-only (no tape); it produces the same
// logits as a full Forward over the prefix.
func (m *Gemma2) DecodeStep(ctx *backend.Context, cache *Gemma2Cache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	x, err := m.embedOne(ctx, token)
	if err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		// Attention sublayer: input_layernorm → capped attention →
		// post_attention_layernorm → add residual.
		xb, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.cappedDecodeAttention(ctx, b, xb, cache, l, pos)
		if err != nil {
			return nil, err
		}
		if a, err = b.PostAttnNorm.Forward(ctx, a); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// FFN sublayer: pre_feedforward_layernorm → GeGLU →
		// post_feedforward_layernorm → add residual.
		xf, err := b.PreFFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if ff, err = b.PostFFNNorm.Forward(ctx, ff); err != nil {
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
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, transpose2D(m.TokEmb))
	if err != nil {
		return nil, err
	}
	// Final-logit soft-cap.
	if cfg.FinalLogitCap > 0 {
		logits, err = exec1(ctx, backend.OpSoftCap, backend.SoftCapAttrs{Cap: cfg.FinalLogitCap}, logits)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// cappedDecodeAttention computes Gemma 2's soft-capped attention for the single new
// query token against the cached keys/values, appending this token's rotated k,v to the
// cache first. It mirrors [Gemma2.cappedAttention]'s per-head numerics — scores·scale
// (1/√QueryPreAttnScalar) → attention-logit soft-cap → softmax over keys → ·V — but for
// ONE query row and WITHOUT the causal mask: a single query legitimately attends to all
// cached keys, so the mask is a no-op for it, and the last-position math is numerically
// identical to the full masked attention. GQA maps query head h to KV head h/(heads/kv).
func (m *Gemma2) cappedDecodeAttention(ctx *backend.Context, b *Gemma2Block, xb *tensor.Tensor, cache *Gemma2Cache, l, pos int) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	hd := cfg.HeadDim
	rep := cfg.Heads / kv

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

	// Pre-softmax score scale as a rank-0 scalar (broadcast over [1,sk]).
	scaleT := tensor.New(tensor.F64, tensor.Shape{})
	scaleT.Storage().F64()[0] = cfg.queryScale()

	heads := make([]*tensor.Tensor, cfg.Heads)
	for h := range cfg.Heads {
		kvHead := h / rep
		qh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * hd, End: (h + 1) * hd}, q)
		if err != nil {
			return nil, err
		}
		kh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: kvHead * hd, End: (kvHead + 1) * hd}, kNew)
		if err != nil {
			return nil, err
		}
		vh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: kvHead * hd, End: (kvHead + 1) * hd}, vNew)
		if err != nil {
			return nil, err
		}
		// scores = qh·khᵀ  [1,sk]
		khT, err := exec1(ctx, backend.OpTranspose, nil, kh)
		if err != nil {
			return nil, err
		}
		scores, err := exec1(ctx, backend.OpMatMul, nil, qh, khT)
		if err != nil {
			return nil, err
		}
		// scores·scale (query_pre_attn_scalar^-0.5)
		if scores, err = exec1(ctx, backend.OpMul, nil, scores, scaleT); err != nil {
			return nil, err
		}
		// attention-logit soft-cap on the scaled scores (no mask for a single query)
		if cfg.AttnLogitCap > 0 {
			if scores, err = exec1(ctx, backend.OpSoftCap, backend.SoftCapAttrs{Cap: cfg.AttnLogitCap}, scores); err != nil {
				return nil, err
			}
		}
		probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
		if err != nil {
			return nil, err
		}
		oh, err := exec1(ctx, backend.OpMatMul, nil, probs, vh) // [1,hd]
		if err != nil {
			return nil, err
		}
		heads[h] = oh
	}
	// Concatenate head outputs → [1, heads·head_dim], then o_proj.
	concat, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, concat, b.Wo)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler
// s, using the KV-cache (one forward per new token). Returns prompt+generated. With a
// greedy sampler the output is identical to argmax-ing a full Forward at each step.
// The decode runs on backend.Default() unless WithBackend overrides it.
func (m *Gemma2) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
