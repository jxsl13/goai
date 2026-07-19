package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LlamaPrefixCache reuses the longest token-exact prefix of the preceding prompt.
// It owns one ordinary contiguous [LlamaCache], so it is suited to sequential chat,
// RAG, and editing workloads that repeatedly run one model with related prompts.
// It is intentionally a single slot: it has no radix tree, LRU policy, paging,
// multi-tenant isolation, or synchronization, and must not be used concurrently.
//
// Call [NewLlamaPrefixCache] to bind a cache to a model, then call Prefill with each
// complete prompt. The returned reuse count is the number of leading prompt tokens
// whose transformer work was retained. The returned tensor is always the one
// next-token row [1,vocab], rather than all per-position prompt logits.
type LlamaPrefixCache struct {
	model  *Llama
	tokens []int
	cache  *LlamaCache
	last   *tensor.Tensor
}

// NewLlamaPrefixCache creates an empty single-slot automatic prefix cache for model.
// The cache retains model by pointer; its weights, geometry, and layer-backend plan
// must therefore not be changed between calls to [LlamaPrefixCache.Prefill].
func NewLlamaPrefixCache(model *Llama) (*LlamaPrefixCache, error) {
	if model == nil {
		return nil, fmt.Errorf("nlp: NewLlamaPrefixCache needs a model")
	}
	if model.TokEmb == nil || model.Config.Heads <= 0 || model.Config.Dim%model.Config.Heads != 0 {
		return nil, fmt.Errorf("nlp: NewLlamaPrefixCache needs an initialized Llama model")
	}
	if err := validateLayerBackends(model.LayerBackends, len(model.Blocks)); err != nil {
		return nil, err
	}
	return &LlamaPrefixCache{model: model, cache: model.NewCache()}, nil
}

// Prefill returns the next-token logits [1,vocab] for the complete prompt and the
// number of leading tokens reused from the preceding call. Reuse is token-exact:
// extensions and divergences retain their exact longest common prefix. Repeating an
// identical prompt reuses every token and returns a copy of the stored last logits.
//
// A strict shortening has no cached logits for its newly exposed last position: to
// avoid retaining a potentially huge [prompt,vocab] tensor, Prefill keeps the first
// len(prompt)-1 tokens and recomputes only the final one. Its reuse count is therefore
// len(prompt)-1. Validation errors (nil context, empty/invalid/over-context prompt, or
// stale model/cache geometry) leave the prior slot unchanged.
func (p *LlamaPrefixCache) Prefill(ctx *backend.Context, prompt []int) (*tensor.Tensor, int, error) {
	if p == nil || p.model == nil || p.cache == nil {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixCache.Prefill needs an initialized cache")
	}
	if ctx == nil {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixCache.Prefill needs a context")
	}
	m := p.model
	if len(prompt) == 0 {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixCache.Prefill needs a non-empty prompt")
	}
	if len(prompt) > m.Config.Ctx {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixCache prompt length %d exceeds context %d", len(prompt), m.Config.Ctx)
	}
	for _, token := range prompt {
		if token < 0 || token >= m.Config.Vocab {
			return nil, 0, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
		}
	}
	prefix, err := validateLlamaPrefixState(m, p.cache)
	if err != nil {
		return nil, 0, err
	}
	if prefix != len(p.tokens) {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixCache token state has %d tokens but KV cache has %d", len(p.tokens), prefix)
	}
	if prefix == 0 {
		if p.last != nil {
			return nil, 0, fmt.Errorf("nlp: empty LlamaPrefixCache has stored logits")
		}
	} else if p.last == nil || p.last.Ndim() != 2 || p.last.Shape()[0] != 1 || p.last.Shape()[1] != m.Config.Vocab {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixCache has invalid stored logits")
	}

	lcp := commonTokenPrefix(p.tokens, prompt)
	if lcp == len(p.tokens) && lcp == len(prompt) {
		return p.last.Cast(p.last.Dtype()), lcp, nil
	}

	keep, reused := lcp, lcp
	if lcp == len(prompt) { // strict shortening: its final-token logits were not retained
		keep = len(prompt) - 1
		reused = keep
	}
	if keep != prefix {
		if err := p.cache.truncate(keep); err != nil {
			return nil, 0, err
		}
	}
	logits, err := m.PrefillAppend(ctx, p.cache, prompt[keep:])
	if err != nil {
		return nil, 0, err
	}
	last, err := logits.Slice(0, logits.Shape()[0]-1, logits.Shape()[0])
	if err != nil {
		return nil, 0, err
	}
	p.tokens = append(p.tokens[:0], prompt...)
	p.last = last.Cast(last.Dtype())
	return last, reused, nil
}

func commonTokenPrefix(a, b []int) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// validateLlamaPrefixState verifies the reusable contiguous-cache contract without
// mutating cache. PrefillAppend and LlamaPrefixCache share it so truncation never
// precedes validation of stale model geometry or malformed K/V state.
func validateLlamaPrefixState(m *Llama, cache *LlamaCache) (int, error) {
	if m == nil {
		return 0, fmt.Errorf("nlp: PrefillAppend needs a model")
	}
	if cache == nil {
		return 0, fmt.Errorf("nlp: PrefillAppend needs a cache")
	}
	if m.TokEmb == nil || m.Config.Heads <= 0 || m.Config.Dim%m.Config.Heads != 0 {
		return 0, fmt.Errorf("nlp: PrefillAppend needs an initialized Llama model")
	}
	if err := validateLayerBackends(m.LayerBackends, len(m.Blocks)); err != nil {
		return 0, err
	}
	if len(cache.K) != len(m.Blocks) || len(cache.V) != len(m.Blocks) {
		return 0, fmt.Errorf("nlp: PrefillAppend cache has %d/%d K/V layers, model has %d", len(cache.K), len(cache.V), len(m.Blocks))
	}
	prefix := cache.Len()
	if cache.NextPos() != prefix {
		return 0, fmt.Errorf("nlp: PrefillAppend needs a contiguous cache from position 0, got %d rows with next position %d", prefix, cache.NextPos())
	}
	kvWidth := m.Config.kvHeads() * (m.Config.Dim / m.Config.Heads)
	for l := range m.Blocks {
		kn, vn := tensorRows(cache.K[l]), tensorRows(cache.V[l])
		if kn != prefix || vn != prefix {
			return 0, fmt.Errorf("nlp: PrefillAppend cache layer %d has %d/%d K/V rows, want %d", l, kn, vn, prefix)
		}
		for _, item := range []struct {
			kind  string
			value *tensor.Tensor
		}{{"key", cache.K[l]}, {"value", cache.V[l]}} {
			if item.value == nil {
				continue
			}
			if item.value.Ndim() != 2 || item.value.Shape()[1] != kvWidth || item.value.Dtype() != m.TokEmb.Dtype() {
				return 0, fmt.Errorf("nlp: PrefillAppend cache layer %d %s must be [%d,%d] %s, got %v %s", l, item.kind, prefix, kvWidth, m.TokEmb.Dtype(), item.value.Shape(), item.value.Dtype())
			}
		}
	}
	return prefix, nil
}
