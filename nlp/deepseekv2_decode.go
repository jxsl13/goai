package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DeepSeekV2Cache holds the per-layer, per-head RECONSTRUCTED key/value tensors accumulated
// during autoregressive DeepSeek-V2 decoding. Unlike a latent-compressed KV-cache (a future
// memory optimization), this caches the fully materialized per-head key concat(k_nope, k_pe)
// [tokens, QKNope+QKRope] and value [tokens, VHead] — exactly the tensors [DeepSeekV2.mlaAttention]
// builds per head in a full Forward — so the decode logits are bit-identical to Forward. K[l][h]
// and V[l][h] index block l, head h; both are nil until the first token. The cache concerns
// only the MLA sublayer; the FFN (dense SwiGLU or sparse DeepSeekMoE) is stateless.
type DeepSeekV2Cache struct {
	K, V [][]*tensor.Tensor // [block][head]; nil until the first token
	bufs []kvBufs           // per-block backing row buffers behind the K, V views (amortized-O(1) append)
}

// NewCache returns an empty KV-cache sized for this model's blocks and heads.
func (m *DeepSeekV2) NewCache() *DeepSeekV2Cache {
	c := &DeepSeekV2Cache{
		K: make([][]*tensor.Tensor, len(m.Blocks)),
		V: make([][]*tensor.Tensor, len(m.Blocks)),
	}
	for l := range m.Blocks {
		c.K[l] = make([]*tensor.Tensor, m.Config.Heads)
		c.V[l] = make([]*tensor.Tensor, m.Config.Heads)
	}
	return c
}

// Len returns the number of tokens currently cached.
func (c *DeepSeekV2Cache) Len() int {
	if len(c.K) == 0 || len(c.K[0]) == 0 || c.K[0][0] == nil {
		return 0
	}
	return c.K[0][0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token]. DeepSeek-V2
// applies no embedding scale.
func (m *DeepSeekV2) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.TokEmb, token, d), nil
}

// DecodeStep advances the DeepSeek-V2 by one token using the KV-cache and returns the
// next-token logits [1,vocab]. pos is the token's absolute position (== cache length before
// the call), used as the decoupled-RoPE offset for q_pe/k_pe. It replays the whole of MLA for
// the single token — the two low-rank latents with their own RMSNorms, the shared rotary key,
// and the rectangular per-head attention — then appends this token's reconstructed per-head
// key concat(k_nope, k_pe) and value to the cache and runs the single query against ALL cached
// keys (no causal mask needed: every cached key is at position ≤ pos). The FFN sublayer is the
// block's own path (dense SwiGLU for the first FirstKDense layers, sparse DeepSeekMoE for the
// rest — the same raw-softmax top-k routing a token would take in Forward). Inference-only (no
// tape); it produces the same logits as a full Forward over the prefix.
func (m *DeepSeekV2) DecodeStep(ctx *backend.Context, cache *DeepSeekV2Cache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	qkHead := cfg.QKNope + cfg.QKRope // per-head query/key width (rectangular vs value)
	kvHead := cfg.QKNope + cfg.VHead  // per-head kv_b output width (k_nope + value)
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: 1, PosOffset: pos}

	// Pre-softmax score scale (rank-0 scalar broadcast over [1,tokens]).
	scaleT := tensor.New(tensor.F64, tensor.Shape{})
	scaleT.Storage().F64()[0] = cfg.softmaxScale()

	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		// Attention sublayer: input_layernorm → MLA → add residual.
		xb, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}

		// Query path: q = q_b_proj(q_a_layernorm(q_a_proj(h))).
		cQ, err := exec1(ctx, backend.OpMatMul, nil, xb, b.WqA)
		if err != nil {
			return nil, err
		}
		if cQ, err = b.QANorm.Forward(ctx, cQ); err != nil {
			return nil, err
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, cQ, b.WqB) // [1, heads·qkHead]
		if err != nil {
			return nil, err
		}

		// KV path: compressed = kv_a_proj_with_mqa(h) → [kv_latent | k_pe]; k_pe is shared.
		compressed, err := exec1(ctx, backend.OpMatMul, nil, xb, b.WkvA)
		if err != nil {
			return nil, err
		}
		kvLatent, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.KVLoraRank}, compressed)
		if err != nil {
			return nil, err
		}
		kPe, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.KVLoraRank, End: cfg.KVLoraRank + cfg.QKRope}, compressed)
		if err != nil {
			return nil, err
		}
		if kvLatent, err = b.KvANorm.Forward(ctx, kvLatent); err != nil {
			return nil, err
		}
		kv, err := exec1(ctx, backend.OpMatMul, nil, kvLatent, b.WkvB) // [1, heads·kvHead]
		if err != nil {
			return nil, err
		}
		// Decoupled RoPE on the SHARED key (one head of width QKRope) at absolute position pos.
		kPeRot, err := exec1(ctx, backend.OpRoPE, rope, kPe)
		if err != nil {
			return nil, err
		}

		heads := make([]*tensor.Tensor, cfg.Heads)
		for h := range cfg.Heads {
			qh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * qkHead, End: (h + 1) * qkHead}, q)
			if err != nil {
				return nil, err
			}
			qNope, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.QKNope}, qh)
			if err != nil {
				return nil, err
			}
			qPe, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.QKNope, End: qkHead}, qh)
			if err != nil {
				return nil, err
			}
			qPeRot, err := exec1(ctx, backend.OpRoPE, rope, qPe)
			if err != nil {
				return nil, err
			}
			queryH, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, qNope, qPeRot) // [1, qkHead]
			if err != nil {
				return nil, err
			}

			kvh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * kvHead, End: (h + 1) * kvHead}, kv)
			if err != nil {
				return nil, err
			}
			kNope, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.QKNope}, kvh)
			if err != nil {
				return nil, err
			}
			valueH, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.QKNope, End: kvHead}, kvh)
			if err != nil {
				return nil, err
			}
			keyH, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, kNope, kPeRot) // [1, qkHead]
			if err != nil {
				return nil, err
			}

			// Append this token's reconstructed per-head key/value to the cache, then attend
			// the single query against ALL cached keys/values.
			if len(cache.bufs) <= l {
				cache.bufs = growSlice(cache.bufs, max(l+1, len(cache.K)))
			}
			kCache, vCache := cache.bufs[l].appendKV(cache.K[l], cache.V[l], h, keyH, valueH)
			cache.K[l][h], cache.V[l][h] = kCache, vCache

			// scores = queryH·kCacheᵀ  [1, tokens]
			kCacheT, err := exec1(ctx, backend.OpTranspose, nil, kCache)
			if err != nil {
				return nil, err
			}
			scores, err := exec1(ctx, backend.OpMatMul, nil, queryH, kCacheT)
			if err != nil {
				return nil, err
			}
			if scores, err = exec1(ctx, backend.OpMul, nil, scores, scaleT); err != nil {
				return nil, err
			}
			// Single query attends all cached keys → no causal mask.
			probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
			if err != nil {
				return nil, err
			}
			oh, err := exec1(ctx, backend.OpMatMul, nil, probs, vCache) // [1, VHead]
			if err != nil {
				return nil, err
			}
			heads[h] = oh
		}
		// Concatenate head outputs → [1, heads·VHead], then o_proj.
		concat, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
		if err != nil {
			return nil, err
		}
		a, err := exec1(ctx, backend.OpMatMul, nil, concat, b.Wo)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}

		// FFN sublayer: post_attention_layernorm → (dense SwiGLU | DeepSeekMoE) → add residual.
		xf, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if b.MoE != nil {
			ff, err = m.moeFFN(ctx, b.MoE, xf)
		} else {
			ff, err = b.Dense.Forward(ctx, xf)
		}
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
	// Untied LM head: logits = hidden · lm_head (stored [dim, vocab]).
	return exec1(ctx, backend.OpMatMul, nil, x, m.LmHead)
}

// Prefill runs the transformer ONCE over the whole prompt, seeds the RECONSTRUCTED
// per-head KV-cache with every prompt token's key/value rows, and returns the full
// logits [seq, vocab] (row seq−1 is the next-token distribution Generate samples
// from). It is the batched replacement for feeding the prompt token-by-token through
// [DeepSeekV2.DecodeStep]: one full-sequence pass through the block stack instead of
// seq single-token dispatch rounds, with a bit-identical cache — the captured
// per-head keys concat(k_nope, k_pe_rotated) and values are the exact tensors
// DecodeStep would have appended (the full-sequence decoupled RoPE rotates row p for
// position p, exactly DecodeStep's PosOffset=p, and the shared k_pe enters every
// head's key identically). The FFN sublayers run the SAME per-layer path as
// DecodeStep — dense SwiGLU or [DeepSeekV2.moeFFN] — which is row-independent, so the
// batched pass reproduces the stepped residual stream bit for bit.
//
// For readers new to LLM inference: generation has two phases. "Prefill" processes
// the whole prompt at once — every token is known up front, so the model can attend
// over the full sequence in a handful of large batched kernels and store each
// layer's keys/values in the cache. "Decode" then produces one token at a time,
// where each new token only needs its own small computation plus the cached past.
// Batching the prefill is the standard first optimization of every serving stack.
//
// cache must be empty (a fresh [DeepSeekV2.NewCache]); Prefill errors on a non-empty
// cache because the full-sequence RoPE and causal mask assume positions 0..seq−1.
// Inference-only, like DecodeStep: run it on a plain backend.NewContext.
func (m *DeepSeekV2) Prefill(ctx *backend.Context, cache *DeepSeekV2Cache, tokens []int) (*tensor.Tensor, error) {
	if n := cache.Len(); n != 0 {
		return nil, fmt.Errorf("nlp: Prefill needs an empty cache, got %d cached tokens", n)
	}
	if len(cache.bufs) < len(m.Blocks) {
		cache.bufs = growSlice(cache.bufs, len(m.Blocks))
	}
	return m.forwardWith(ctx, tokens, func(l int, b *DeepSeekV2Block, xb *tensor.Tensor, seq int) (*tensor.Tensor, error) {
		return m.mlaAttention(ctx, b, xb, seq, func(h int, keyH, valueH *tensor.Tensor) {
			// One multi-row Append per layer per head: the whole prompt's per-head
			// [seq, qkHead]/[seq, VHead] rows land in the cache's row buffers in a
			// single typed block copy each.
			kCache, vCache := cache.bufs[l].appendKV(cache.K[l], cache.V[l], h, keyH, valueH)
			cache.K[l][h], cache.V[l][h] = kCache, vCache
		})
	})
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler s,
// using the KV-cache (one batched [DeepSeekV2.Prefill] over the prompt, then one
// [DeepSeekV2.DecodeStep] per new token). Returns prompt+generated. With a greedy
// sampler the output is identical to argmax-ing a full Forward at each step. The decode runs
// on backend.Default() unless WithBackend overrides it.
func (m *DeepSeekV2) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
