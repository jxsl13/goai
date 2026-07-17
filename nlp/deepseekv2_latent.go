package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DeepSeekV2LatentCache is the ABSORBED-MLA latent KV-cache — the memory win Multi-head
// Latent Attention was designed for (DeepSeek-AI, arXiv:2405.04434 §2.1.2). Where
// [DeepSeekV2Cache] stores the RECONSTRUCTED per-head tensors — per token, per layer,
// Heads·(QKNope+QKRope) key values plus Heads·VHead value values — this cache stores only
// what MLA actually needs to remember:
//
//   - CKV[l]: the normalized kv latent c_KV = kv_a_layernorm(kv_latent), KVLoraRank values
//     per token — the SHARED low-rank code every head's key AND value decompress from.
//   - KPE[l]: the shared decoupled rotary key k_pe (post-RoPE), QKRope values per token —
//     one head's worth, shared by all heads (MQA-style).
//
// That is KVLoraRank+QKRope values per token per layer instead of
// Heads·(QKNope+QKRope+VHead) — e.g. 16+8 = 24 vs 4·(16+8+16) = 160 at the golden-model
// geometry, a 6.67× KV-memory reduction (far larger at DeepSeek-V2 236B scale: 576 vs
// 128·(128+64+128) = 40960, ~71×).
//
// The naive way to decode from a latent cache — re-expand every cached token through
// kv_b_proj each step — costs O(T) extra matmul work per step, O(T²) over a decode.
// [DeepSeekV2.DecodeStepLatent] instead uses the WEIGHT-ABSORPTION identity: because
// k_nope_t,h = c_KV_t · W_K,h (the K-half column block of kv_b_proj) and
// value_t,h = c_KV_t · W_V,h are linear in the latent,
//
//	content score:  q_nope_h · k_nope_t,hᵀ = (q_nope_h · W_K,hᵀ) · c_KV_tᵀ
//	head output:    Σ_t w_t · value_t,h    = (w · CKV) · W_V,h
//
// so W_K,h can be "absorbed" into the query (one [1,QKNope]×[QKNope,KVLoraRank] matmul per
// head PER STEP, not per cached token) and W_V,h into the attention-weighted latent (one
// [1,KVLoraRank]×[KVLoraRank,VHead] matmul per head per step). The cache is only ever
// touched through two skinny matmuls — scores via [1,KVLoraRank]×[KVLoraRank,T] and the
// context latent via [1,T]×[T,KVLoraRank] — the same O(T) per step as the reconstructed
// path, on a cache 6.67× smaller. The absorbed per-head blocks of kv_b_proj (wkT, wv) are
// split out of WkvB once in [DeepSeekV2.NewLatentCache].
//
// CKV[l] and KPE[l] are nil until the first token and grow via amortized-O(1) rowBufs.
// The cache concerns only the MLA sublayer; the FFN is stateless.
type DeepSeekV2LatentCache struct {
	CKV []*tensor.Tensor // [block] → [tokens, KVLoraRank] normalized kv latent
	KPE []*tensor.Tensor // [block] → [tokens, QKRope] shared rotary key (post-RoPE)
	// bufs backs the CKV, KPE views with amortized-O(1) row appends (k rowBufs hold the
	// latents, v rowBufs the rotary keys — the pairing is positional, not semantic).
	bufs kvBufs

	// Absorbed kv_b_proj blocks, precomputed once per cache from the model's WkvB
	// [KVLoraRank, Heads·(QKNope+VHead)]:
	wkT [][]*tensor.Tensor // [block][head] (K-half columns)ᵀ [QKNope, KVLoraRank]
	wv  [][]*tensor.Tensor // [block][head] V-half columns [KVLoraRank, VHead]
}

// NewLatentCache returns an empty absorbed-MLA latent cache for this model, precomputing
// the per-block, per-head absorbed column blocks of kv_b_proj: wkT[l][h] is the transpose
// of WkvB's k_nope columns for head h (so q_nope·wkT lands the query in latent space) and
// wv[l][h] is WkvB's value columns for head h (so the attention-weighted latent expands to
// the head output). This is a cheap host-side slice+transpose of already-loaded weights —
// O(KVLoraRank·(QKNope+VHead)) per head — done once per cache, never per step.
func (m *DeepSeekV2) NewLatentCache() *DeepSeekV2LatentCache {
	cfg := m.Config
	kvHead := cfg.QKNope + cfg.VHead
	c := &DeepSeekV2LatentCache{
		CKV: make([]*tensor.Tensor, len(m.Blocks)),
		KPE: make([]*tensor.Tensor, len(m.Blocks)),
		wkT: make([][]*tensor.Tensor, len(m.Blocks)),
		wv:  make([][]*tensor.Tensor, len(m.Blocks)),
	}
	for l, b := range m.Blocks {
		c.wkT[l] = make([]*tensor.Tensor, cfg.Heads)
		c.wv[l] = make([]*tensor.Tensor, cfg.Heads)
		for h := range cfg.Heads {
			kBlock, err := b.WkvB.Slice(1, h*kvHead, h*kvHead+cfg.QKNope)
			if err != nil { // WkvB geometry fixed at load; unreachable
				panic(err)
			}
			vBlock, err := b.WkvB.Slice(1, h*kvHead+cfg.QKNope, (h+1)*kvHead)
			if err != nil {
				panic(err)
			}
			c.wkT[l][h] = transpose2D(kBlock) // [QKNope, KVLoraRank]
			c.wv[l][h] = vBlock.Contiguous()  // [KVLoraRank, VHead]
		}
	}
	return c
}

// Len returns the number of tokens currently cached.
func (c *DeepSeekV2LatentCache) Len() int {
	if len(c.CKV) == 0 || c.CKV[0] == nil {
		return 0
	}
	return c.CKV[0].Shape()[0]
}

// DecodeStepLatent advances the DeepSeek-V2 by one token using the absorbed-MLA latent
// cache and returns the next-token logits [1,vocab]. pos is the token's absolute position
// (== cache length before the call), the decoupled-RoPE offset for q_pe/k_pe.
//
// It computes the same MLA as [DeepSeekV2.DecodeStep] but never materializes per-head keys
// or values for the cached tokens. Per block it appends this token's normalized kv latent
// c_KV [KVLoraRank] and shared post-RoPE k_pe [QKRope] to the cache, then per head:
//
//	q_lat  = q_nope · wkTₕ                     [1,KVLoraRank]  (absorb W_K into the query)
//	scores = (q_lat·CKVᵀ + q_pe·KPEᵀ) · scale  [1,tokens]      (content + rope halves)
//	w      = softmax(scores)                                   (no mask: all keys ≤ pos)
//	c_att  = w · CKV                           [1,KVLoraRank]  (attention in latent space)
//	out_h  = c_att · wvₕ                       [1,VHead]       (absorb W_V into the output)
//
// This is algebraically the reconstructed attention with the matmuls reassociated —
// q_nope·(CKV·W_K)ᵀ = (q_nope·W_Kᵀ)·CKVᵀ and (w·CKV)·W_V = w·(CKV·W_V) — so logits match
// Forward and DecodeStep up to float reassociation (~1e-16 observed at golden dims). The FFN
// sublayer is the block's own path (dense SwiGLU or DeepSeekMoE), identical to DecodeStep.
// Inference-only (no tape).
func (m *DeepSeekV2) DecodeStepLatent(ctx *backend.Context, cache *DeepSeekV2LatentCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	cfg := m.Config
	qkHead := cfg.QKNope + cfg.QKRope // per-head query width (nope + rotary)
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: 1, PosOffset: pos}

	// Pre-softmax score scale (rank-0 scalar broadcast over [1,tokens]).
	scaleT := tensor.New(tensor.F64, tensor.Shape{})
	scaleT.Storage().F64()[0] = cfg.softmaxScale()

	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		// Attention sublayer: input_layernorm → absorbed MLA → add residual.
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

		// KV path: compressed = kv_a_proj_with_mqa(h) → [kv_latent | k_pe]. ONLY the
		// normalized latent and the rotated shared key are cached — kv_b_proj is never
		// applied to cached tokens.
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
		cKV, err := b.KvANorm.Forward(ctx, kvLatent) // [1, KVLoraRank]
		if err != nil {
			return nil, err
		}
		kPeRot, err := exec1(ctx, backend.OpRoPE, rope, kPe) // [1, QKRope] at position pos
		if err != nil {
			return nil, err
		}

		// Append this token's latent + rotary key; refresh the public views.
		ckvAll, kpeAll := cache.bufs.appendKV(cache.CKV, cache.KPE, l, cKV, kPeRot)
		cache.CKV[l], cache.KPE[l] = ckvAll, kpeAll

		// Transposed caches for the score matmuls, built ONCE per block per step (O(tokens),
		// not per head).
		ckvT, err := exec1(ctx, backend.OpTranspose, nil, ckvAll) // [KVLoraRank, tokens]
		if err != nil {
			return nil, err
		}
		kpeT, err := exec1(ctx, backend.OpTranspose, nil, kpeAll) // [QKRope, tokens]
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

			// Absorb W_K into the query: q_lat = q_nope · (W_K,h)ᵀ — per step, not per token.
			qLat, err := exec1(ctx, backend.OpMatMul, nil, qNope, cache.wkT[l][h]) // [1, KVLoraRank]
			if err != nil {
				return nil, err
			}
			// Content scores directly against the latent cache; rope scores against the
			// shared rotary cache.
			scoreC, err := exec1(ctx, backend.OpMatMul, nil, qLat, ckvT) // [1, tokens]
			if err != nil {
				return nil, err
			}
			scoreP, err := exec1(ctx, backend.OpMatMul, nil, qPeRot, kpeT) // [1, tokens]
			if err != nil {
				return nil, err
			}
			scores, err := exec1(ctx, backend.OpAdd, nil, scoreC, scoreP)
			if err != nil {
				return nil, err
			}
			if scores, err = exec1(ctx, backend.OpMul, nil, scores, scaleT); err != nil {
				return nil, err
			}
			// Single query attends all cached positions → no causal mask.
			probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
			if err != nil {
				return nil, err
			}
			// Attention in latent space, then absorb W_V into the output.
			ctxLat, err := exec1(ctx, backend.OpMatMul, nil, probs, ckvAll) // [1, KVLoraRank]
			if err != nil {
				return nil, err
			}
			oh, err := exec1(ctx, backend.OpMatMul, nil, ctxLat, cache.wv[l][h]) // [1, VHead]
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
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
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
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Untied LM head: logits = hidden · lm_head (stored [dim, vocab]).
	return exec1(ctx, backend.OpMatMul, nil, x, m.LmHead)
}

// GenerateLatent autoregressively decodes up to maxNew tokens after prompt with the sampler
// s, using the absorbed-MLA latent KV-cache — the same decoding as [DeepSeekV2.Generate]
// but holding (KVLoraRank+QKRope) instead of Heads·(QKNope+QKRope+VHead) values per token
// per layer. Returns prompt+generated. The decode runs on backend.Default() unless
// WithBackend overrides it.
func (m *DeepSeekV2) GenerateLatent(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	if gc.be != nil {
		ctx = ctx.WithBackend(gc.be)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: GenerateLatent needs a non-empty prompt")
	}
	cache := m.NewLatentCache()
	out := append([]int(nil), prompt...)

	var logits *tensor.Tensor
	pos := 0
	for _, tok := range prompt {
		l, err := m.DecodeStepLatent(ctx, cache, tok, pos)
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
		l, err := m.DecodeStepLatent(ctx, cache, next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}
