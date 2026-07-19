package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// layerSkipDecodeTrace is a test-only observer carried through the real KVQ path.
// A nil trace costs only one branch per executed block.
type layerSkipDecodeTrace struct {
	blockTokens []int
	rounds      []layerSkipRoundTrace
	cache       *LlamaCache
}

type layerSkipRoundTrace struct {
	before, drafted, emitted, after int
	layerRows                       []int
}

func tensorRows(t *tensor.Tensor) int {
	if t == nil {
		return 0
	}
	return t.Shape()[0]
}

// truncate rolls speculative rows back. Unlike eviction, this really rewinds the
// stream: rejected tokens were never committed, so kvPos needs no dropped-row bump.
func (c *LlamaCache) truncate(n int) error {
	if c == nil {
		return fmt.Errorf("nlp: cannot truncate a nil LlamaCache")
	}
	if n < 0 {
		return fmt.Errorf("nlp: cannot truncate LlamaCache to %d rows", n)
	}
	if len(c.K) != len(c.V) {
		return fmt.Errorf("nlp: LlamaCache has %d key layers and %d value layers", len(c.K), len(c.V))
	}
	c.bufs.k = growSlice(c.bufs.k, len(c.K))
	c.bufs.v = growSlice(c.bufs.v, len(c.V))
	for l := range c.K {
		kn, vn := tensorRows(c.K[l]), tensorRows(c.V[l])
		if kn != vn {
			return fmt.Errorf("nlp: LlamaCache layer %d has %d key rows and %d value rows", l, kn, vn)
		}
		if n > kn {
			return fmt.Errorf("nlp: cannot truncate LlamaCache layer %d from %d to %d rows", l, kn, n)
		}
		c.K[l] = c.bufs.k[l].Truncate(c.K[l], n)
		c.V[l] = c.bufs.v[l].Truncate(c.V[l], n)
	}
	return nil
}

// cachedBlockRange executes blocks [start,end) over rows whose first absolute
// position is pos, appending each block's new K/V to its potentially shorter cache.
// Causal OpMHA interprets rectangular [rows,past+rows] attention with offset
// sk-sq=past, exactly the verification-window mask LayerSkip needs.
func (m *Llama) cachedBlockRange(
	ctx *backend.Context,
	cache *LlamaCache,
	x *tensor.Tensor,
	start, end, pos int,
	trace *layerSkipDecodeTrace,
) (*tensor.Tensor, error) {
	rows := x.Shape()[0]
	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{
		Heads: cfg.Heads, KVHeads: kv, Causal: rows > 1, Scale: cfg.attnScale(),
	}
	return m.blockRange(ctx, x, start, end, func(layerCtx *backend.Context, l int, q, k, v *tensor.Tensor) (*tensor.Tensor, error) {
		if kn, vn := tensorRows(cache.K[l]), tensorRows(cache.V[l]); kn != pos || vn != pos {
			return nil, fmt.Errorf("nlp: LayerSkip cache layer %d starts verification at position %d with %d/%d K/V rows", l, pos, kn, vn)
		}
		var err error
		q, err = exec1(layerCtx, backend.OpRoPE, backend.RoPEAttrs{
			Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos,
		}, q)
		if err != nil {
			return nil, err
		}
		k, err = exec1(layerCtx, backend.OpRoPE, backend.RoPEAttrs{
			Base: cfg.RopeBase, Heads: kv, PosOffset: pos,
		}, k)
		if err != nil {
			return nil, err
		}
		cache.K[l], cache.V[l] = cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		if trace != nil {
			trace.blockTokens[l] += rows
		}
		return exec1(layerCtx, backend.OpMHA, attn, q, cache.K[l], cache.V[l])
	})
}

// layerSkipDraftStep processes one token through the early block prefix and returns
// the PRE-final-norm exit residual. It advances only the early layers' KV rows.
func (m *Llama) layerSkipDraftStep(
	ctx *backend.Context,
	cache *LlamaCache,
	token, pos, exitLayer int,
	trace *layerSkipDecodeTrace,
) (*tensor.Tensor, error) {
	if err := cache.pos.admit(cache.Len(), pos, "LlamaCache"); err != nil {
		return nil, err
	}
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	if x, err = scaleScalar(ctx, x, m.Config.EmbeddingMult); err != nil {
		return nil, err
	}
	return m.cachedBlockRange(ctx, cache, x, 0, exitLayer+1, pos, trace)
}

// llamaSelfSpeculativeDecodeKVQ implements LayerSkip Appendix A.6: early layers
// draft into one shared KV cache and an exit-query buffer; remaining layers verify
// that buffer in one rectangular causal pass, then all layer caches roll back to the
// committed prefix minus its last (not-yet-processed) token.
func llamaSelfSpeculativeDecodeKVQ(
	model *Llama,
	prompt []int,
	maxNew, exitLayer, lookahead int,
	s *Sampler,
	trace *layerSkipDecodeTrace,
) ([]int, SpecStats, error) {
	const op = "LlamaSelfSpeculativeDecode"
	var stats SpecStats
	if s == nil {
		return nil, stats, fmt.Errorf("nlp: %s needs a sampler", op)
	}
	if len(prompt) == 0 {
		return nil, stats, fmt.Errorf("nlp: %s needs a non-empty prompt", op)
	}
	if len(prompt) > model.Config.Ctx {
		return nil, stats, fmt.Errorf("nlp: %s prompt length %d exceeds context %d", op, len(prompt), model.Config.Ctx)
	}
	for _, tok := range prompt {
		if tok < 0 || tok >= model.Config.Vocab {
			return nil, stats, fmt.Errorf("nlp: token %d outside vocab %d", tok, model.Config.Vocab)
		}
	}
	if lookahead < 1 {
		lookahead = 4
	}

	ctx := backend.NewContext()
	cache := model.NewCache()
	out := append([]int(nil), prompt...)
	if trace != nil {
		trace.blockTokens = make([]int, len(model.Blocks))
		trace.cache = cache
	}
	if maxNew <= 0 || len(out) == model.Config.Ctx {
		return out, stats, nil
	}
	if err := validateLayerBackends(model.LayerBackends, len(model.Blocks)); err != nil {
		return nil, stats, err
	}
	// The cache intentionally stops one token before the committed sequence. That
	// token is the first input of every speculative window (and of final-slot fallback).
	if len(prompt) > 1 {
		if _, err := model.Prefill(ctx, cache, prompt[:len(prompt)-1]); err != nil {
			return nil, stats, err
		}
	}

	for generated := 0; generated < maxNew && len(out) < model.Config.Ctx; {
		remainingCtx := model.Config.Ctx - len(out)
		if remainingCtx == 1 {
			logits, err := model.DecodeStep(ctx, cache, out[len(out)-1], len(out)-1)
			if err != nil {
				return nil, stats, err
			}
			out = append(out, sampleDist(s.Dist(rowAt(logits, 0)), s.source()))
			generated++
			continue
		}

		k := min(lookahead, remainingCtx-1)
		k = min(k, maxNew-generated)
		base := len(out) - 1
		if got := cache.NextPos(); got != base {
			return nil, stats, fmt.Errorf("nlp: LayerSkip cache next position %d, want committed prefix minus one at %d", got, base)
		}

		qs := make([][]float64, 0, k)
		drafts := make([]int, 0, k)
		var queryBuf rowBuf
		var queries *tensor.Tensor
		token := out[len(out)-1]
		for i := range k {
			h, err := model.layerSkipDraftStep(ctx, cache, token, base+i, exitLayer, trace)
			if err != nil {
				return nil, stats, err
			}
			queries = queryBuf.Append(queries, h)
			logits, err := model.Unembed(ctx, h)
			if err != nil {
				return nil, stats, err
			}
			q := s.Dist(rowAt(logits, 0))
			token = sampleDist(q, s.source())
			qs = append(qs, q)
			drafts = append(drafts, token)
		}
		// The last draft has no q distribution, but its early residual is the
		// all-accepted bonus query p_K and its early K/V must enter the shared cache.
		h, err := model.layerSkipDraftStep(ctx, cache, token, base+k, exitLayer, trace)
		if err != nil {
			return nil, stats, err
		}
		queries = queryBuf.Append(queries, h)

		verified, err := model.cachedBlockRange(ctx, cache, queries, exitLayer+1, len(model.Blocks), base, trace)
		if err != nil {
			return nil, stats, err
		}
		logits, err := model.Unembed(ctx, verified)
		if err != nil {
			return nil, stats, err
		}
		ps := make([][]float64, 0, k+1)
		for i := 0; i <= k; i++ {
			ps = append(ps, s.Dist(rowAt(logits, i)))
		}

		accepted := SpeculativeRun(ps, qs, drafts, s.source())
		stats.Proposed += k
		stats.Accepted += len(accepted) - 1
		before := len(out)
		for _, tok := range accepted {
			out = append(out, tok)
			generated++
			if generated >= maxNew || len(out) >= model.Config.Ctx {
				break
			}
		}
		if err := cache.truncate(len(out) - 1); err != nil {
			return nil, stats, err
		}
		if trace != nil {
			rows := make([]int, len(cache.K))
			for l := range rows {
				rows[l] = tensorRows(cache.K[l])
			}
			trace.rounds = append(trace.rounds, layerSkipRoundTrace{
				before: before, drafted: k, emitted: len(out) - before, after: len(out), layerRows: rows,
			})
		}
	}
	return out, stats, nil
}
