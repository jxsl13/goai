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
	bufs kvBufs           // backing row buffers behind the K, V views (amortized-O(1) append)
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
	x := embedRow(m.TokEmb, token, d)
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
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
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
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Tied LM head: logits = hidden · embedᵀ (unscaled table).
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.tiedHead())
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
	kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
	cache.K[l], cache.V[l] = kNew, vNew

	// Pre-softmax score scale as a rank-0 scalar (broadcast over [1,sk]).
	scaleT := tensor.New(tensor.F64, tensor.Shape{})
	scaleT.Storage().F64()[0] = cfg.queryScale()

	heads := make([]*tensor.Tensor, cfg.Heads)
	//perfscan:ignore PS5001 h/rep integer div feeds slice index (unsafe to hoist); one div/head trivial
	for h := range cfg.Heads {
		kvHead := h / rep
		//perfscan:ignore PS6017 exec1 OpSlice per head; decode attention matmul-dominated
		qh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * hd, End: (h + 1) * hd}, q)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 exec1 OpSlice per head; matmul-dominated decode
		kh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: kvHead * hd, End: (kvHead + 1) * hd}, kNew)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 exec1 OpSlice per head; matmul-dominated decode
		vh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: kvHead * hd, End: (kvHead + 1) * hd}, vNew)
		if err != nil {
			return nil, err
		}
		// scores = qh·khᵀ  [1,sk]
		//perfscan:ignore PS6017 exec1 OpTranspose per head; op-dominated decode
		khT, err := exec1(ctx, backend.OpTranspose, nil, kh)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 exec1 OpMatMul (scores) per head; matmul-dominated
		scores, err := exec1(ctx, backend.OpMatMul, nil, qh, khT)
		if err != nil {
			return nil, err
		}
		// scores·scale (query_pre_attn_scalar^-0.5)
		//perfscan:ignore PS6017 exec1 OpMul scale per head; op-dominated, negligible
		if scores, err = exec1(ctx, backend.OpMul, nil, scores, scaleT); err != nil {
			return nil, err
		}
		// attention-logit soft-cap on the scaled scores (no mask for a single query)
		if cfg.AttnLogitCap > 0 {
			//perfscan:ignore PS6016,PS6017 SoftCapAttrs literal per head; resource-only, negligible | exec1 OpSoftCap per head; op-dominated decode
			if scores, err = exec1(ctx, backend.OpSoftCap, backend.SoftCapAttrs{Cap: cfg.AttnLogitCap}, scores); err != nil {
				return nil, err
			}
		}
		//perfscan:ignore PS6017 exec1 OpSoftmax per head; op-dominated decode
		probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 exec1 OpMatMul (probs.v) per head; matmul-dominated
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

// Prefill runs the transformer ONCE over the whole prompt, seeds the KV-cache with
// every prompt token's key/value rows, and returns the full logits [seq, vocab]
// (row seq−1 is the next-token distribution Generate samples from). It is the
// batched replacement for feeding the prompt token-by-token through
// [Gemma2.DecodeStep]: one full-sequence pass through the block stack instead of
// seq single-token dispatch rounds, with a bit-identical cache. The captured rows
// are exactly what cappedDecodeAttention appends per token — the full-width
// POST-RoPE k and the raw (unrotated) post-projection v [seq, kvHeads·headDim],
// tapped in cappedAttention BEFORE its per-head slicing — and the full-sequence
// RoPE rotates row p for position p, exactly DecodeStep's PosOffset=p. The
// soft-capped causal attention over rows 0..p equals the single-query decode
// attention at row p, so decoding may continue from the seeded cache as if the
// prompt had been stepped through one token at a time.
//
// For readers new to LLM inference: generation has two phases. "Prefill" processes
// the whole prompt at once — every token is known up front, so the model can attend
// over the full sequence in a handful of large batched kernels and store each
// layer's keys/values in the cache. "Decode" then produces one token at a time,
// where each new token only needs its own small computation plus the cached past.
// Batching the prefill is the standard first optimization of every serving stack.
//
// cache must be empty (a fresh [Gemma2.NewCache]); Prefill errors on a non-empty
// cache because the full-sequence RoPE and causal mask assume positions 0..seq−1.
// Inference-only, like DecodeStep: run it on a plain backend.NewContext.
func (m *Gemma2) Prefill(ctx *backend.Context, cache *Gemma2Cache, tokens []int) (*tensor.Tensor, error) {
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
// s, using the KV-cache (one batched [Gemma2.Prefill] over the prompt, then one
// [Gemma2.DecodeStep] per new token). Returns prompt+generated. With a
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
