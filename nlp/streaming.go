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
//
// Only the attention core above is specific to streaming: the surrounding block runs
// through the shared [Llama.blockStack], so a Qwen2's projection biases, a Qwen3's
// QK-norm and Granite's four scalars are applied exactly as [Llama.DecodeStep] and
// Forward apply them. (They were NOT, before this path was routed through the shared
// stack — this function carried its own copy of the block body, written before those
// hooks existed, so streaming a Qwen2/Qwen3/Granite model silently produced different
// logits than the non-streaming path. TestStreamGenerateMatchesGenerate is the gate.)
//
// The cached k,v are post-bias and post-QK-norm but pre-RoPE, which is the correct
// place to bound them: both hooks are position-independent, so applying them once
// before caching is identical to re-applying them to the whole cache every step.
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
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false, Scale: cfg.attnScale()}
	h, err := m.blockStack(ctx, x, func(layerCtx *backend.Context, l int, q, k, v *tensor.Tensor) (*tensor.Tensor, error) {
		// append the RAW (pre-RoPE) k,v and bound the cache to sinks + recent window
		cache.K[l] = keepSinkRecent(concatRows(cache.K[l], k), sinks, window)
		cache.V[l] = keepSinkRecent(concatRows(cache.V[l], v), sinks, window)
		n := cache.K[l].Shape()[0]
		// RoPE using cache positions: keys at 0..n−1, the query (last entry) at n−1
		qr, err := exec1(layerCtx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: n - 1}, q)
		if err != nil {
			return nil, err
		}
		kr, err := exec1(layerCtx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, cache.K[l])
		if err != nil {
			return nil, err
		}
		return exec1(layerCtx, backend.OpMHA, attn, qr, kr, cache.V[l])
	})
	if err != nil {
		return nil, err
	}
	// Final RMSNorm + output projection + the Granite logits scaling.
	return m.Unembed(ctx, h)
}

// StreamGenerate decodes up to maxNew tokens after prompt with StreamingLLM's bounded
// cache (sinks attention-sink tokens + a rolling window), so it runs at constant memory
// regardless of how long the stream grows. Unlike KV-cache Generate it is not bounded by
// the model's context length. Returns prompt+generated.
//
// Relationship to [Llama.Generate]: while the budget covers the whole sequence
// (sinks+window ≥ len(prompt)+maxNew) no eviction happens and this returns the SAME
// tokens as Generate for the same model, prompt and greedy sampler — the two are one
// forward pass expressed over two cache layouts, and TestStreamGenerateMatchesGenerate
// holds them to it on Qwen2-, Qwen3- and Granite-shaped models. Past the budget the two
// legitimately diverge: evicting the middle of the cache is the whole point, and the
// output is an approximation of full attention, not an equal of it.
//
// Note that this runs one extra [Llama.StreamStep] after the final sampled token (whose
// logits are discarded), so a stream of n tokens costs n+1 steps. Stopping early on
// [WithEOS] skips that trailing step, since nothing consumes its logits.
func (m *Llama) StreamGenerate(prompt []int, maxNew, sinks, window int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: StreamGenerate needs a non-empty prompt")
	}
	var gc genConfig
	for _, o := range opts {
		o(&gc)
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
		if gc.stopEOS(next, s) {
			break
		}
		if logits, err = m.StreamStep(ctx, cache, next, sinks, window); err != nil {
			return nil, err
		}
	}
	return out, nil
}
