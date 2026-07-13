package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// StreamCache is a bounded KV-cache for StreamingLLM decoding (Xiao et al. 2023,
// "Efficient Streaming Language Models with Attention Sinks", arXiv:2309.17453, §R97).
// Unlike LlamaCache it keeps at most sinks+window entries per layer and stores keys
// BEFORE the rotary embedding, so RoPE can be re-applied at each step using each
// token's position WITHIN the current cache — the trick that lets generation run far
// past the training context length at constant memory.
type StreamCache struct {
	K, V []*tensor.Tensor // per block, PRE-RoPE, ≤ sinks+window rows each
}

// NewStreamCache returns an empty StreamingLLM cache sized for this model's blocks.
func (m *Llama) NewStreamCache() *StreamCache {
	return &StreamCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached.
func (c *StreamCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// keepSinkRecent bounds a [rows,d] cache to the first sinks rows (the attention sinks)
// plus the last window rows (the rolling recent window), evicting the middle. It is a
// no-op while rows ≤ sinks+window.
func keepSinkRecent(t *tensor.Tensor, sinks, window int) *tensor.Tensor {
	rows, d := t.Shape()[0], t.Shape()[1]
	if rows <= sinks+window {
		return t
	}
	out := tensor.New(t.Dtype(), tensor.Shape{sinks + window, d})
	for i := range sinks {
		for j := range d {
			out.SetF64(t.AtF64(i, j), i, j)
		}
	}
	for i := range window {
		for j := range d {
			out.SetF64(t.AtF64(rows-window+i, j), sinks+i, j)
		}
	}
	return out
}

// StreamStep advances the model by one token with the bounded StreamingLLM cache and
// returns the next-token logits [1,vocab]. It keeps the first `sinks` "attention sink"
// tokens (default 4 in the paper) and a rolling window of the last `window` tokens,
// evicting the middle. Keys/values are cached pre-RoPE; each step re-applies RoPE using
// positions within the current cache (0..cacheLen−1), so relative distances stay bounded
// by sinks+window and generation never runs out of positional range. Inference-only.
func (m *Llama) StreamStep(ctx *backend.Context, cache *StreamCache, token, sinks, window int) (*tensor.Tensor, error) {
	if sinks < 0 || window < 1 {
		return nil, fmt.Errorf("nlp: StreamStep needs sinks≥0 and window≥1, got %d/%d", sinks, window)
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
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
		// append the RAW (pre-RoPE) k,v and bound the cache to sinks + recent window
		cache.K[l] = keepSinkRecent(concatRows(cache.K[l], k), sinks, window)
		cache.V[l] = keepSinkRecent(concatRows(cache.V[l], v), sinks, window)
		n := cache.K[l].Shape()[0]
		// RoPE using cache positions: keys at 0..n−1, the query (last entry) at n−1
		qr, err := exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: n - 1}, q)
		if err != nil {
			return nil, err
		}
		kr, err := exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, cache.K[l])
		if err != nil {
			return nil, err
		}
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false}, qr, kr, cache.V[l])
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
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// StreamGenerate decodes up to maxNew tokens after prompt with StreamingLLM's bounded
// cache (sinks attention-sink tokens + a rolling window), so it runs at constant memory
// regardless of how long the stream grows. Unlike KV-cache Generate it is not bounded by
// the model's context length. Returns prompt+generated.
func (m *Llama) StreamGenerate(prompt []int, maxNew, sinks, window int, s TokenSampler) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: StreamGenerate needs a non-empty prompt")
	}
	ctx := backend.NewContext()
	cache := m.NewStreamCache()
	out := append([]int(nil), prompt...)

	var logits *tensor.Tensor
	var err error
	for _, tok := range prompt {
		if logits, err = m.StreamStep(ctx, cache, tok, sinks, window); err != nil {
			return nil, err
		}
	}
	for range maxNew {
		next := s.SampleWithHistory(rowLogits(logits), out)
		out = append(out, next)
		if logits, err = m.StreamStep(ctx, cache, next, sinks, window); err != nil {
			return nil, err
		}
	}
	return out, nil
}
