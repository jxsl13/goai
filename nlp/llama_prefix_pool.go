package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LlamaPrefixPool retains a bounded least-recently-used set of complete prompt
// states. For each request it finds the longest token-exact prefix among entries
// with the same cache salt, copies that contiguous K/V prefix into a new slot, and
// computes only the remaining suffix through [Llama.PrefillAppend]. An empty salt
// is a shared default namespace; distinct non-empty salts isolate reuse between
// trust groups, including the cache-hit timing signal.
//
// This is a small in-process, sequential cache for repeated chat/RAG prompts. A
// salt-separated compressed radix index finds the matching complete entry, but
// each entry still owns a complete contiguous K/V cache: entries do not physically
// share blocks, and the pool has no RadixAttention kernel, hash-block table,
// reference counts, paging, scheduler, or synchronization. It must not be used
// concurrently.
type LlamaPrefixPool struct {
	model     *Llama
	capacity  int
	slots     []llamaPrefixPoolSlot
	spare     *LlamaCache
	clock     uint64
	roots     map[string]*llamaPrefixRadixNode
	radixFree []*llamaPrefixRadixNode
	lru       int
	mru       int
}

type llamaPrefixPoolSlot struct {
	salt   string
	tokens []int
	cache  *LlamaCache
	last   *tensor.Tensor
	used   uint64
	prev   int
	next   int
}

// NewLlamaPrefixPool creates a salt-isolated prefix pool holding at most capacity
// complete prompt states. Capacity must be positive. The pool retains model by
// pointer; do not change its weights, geometry, or layer-backend plan while using it.
func NewLlamaPrefixPool(model *Llama, capacity int) (*LlamaPrefixPool, error) {
	if model == nil {
		return nil, fmt.Errorf("nlp: NewLlamaPrefixPool needs a model")
	}
	if capacity <= 0 {
		return nil, fmt.Errorf("nlp: NewLlamaPrefixPool capacity must be positive, got %d", capacity)
	}
	if model.TokEmb == nil || model.Config.Heads <= 0 || model.Config.Dim%model.Config.Heads != 0 {
		return nil, fmt.Errorf("nlp: NewLlamaPrefixPool needs an initialized Llama model")
	}
	if err := validateLayerBackends(model.LayerBackends, len(model.Blocks)); err != nil {
		return nil, err
	}
	return &LlamaPrefixPool{
		model: model, capacity: capacity, roots: make(map[string]*llamaPrefixRadixNode),
		radixFree: make([]*llamaPrefixRadixNode, 0, 4), lru: -1, mru: -1,
	}, nil
}

// Len returns the number of complete prompt states currently retained. It is at
// most the capacity passed to [NewLlamaPrefixPool].
func (p *LlamaPrefixPool) Len() int {
	if p == nil {
		return 0
	}
	return len(p.slots)
}

// Prefill returns next-token logits [1,vocab] for prompt and the number of leading
// tokens whose transformer work was reused from a retained entry with cacheSalt.
// An exact prompt hit reuses every token. A strict shortening recomputes only its
// newly exposed final token because slots retain one final logit row, not the much
// larger [prompt,vocab] tensor.
//
// A successful non-exact request inserts a new most-recently-used entry and evicts
// the least-recently-used entry when the pool is full. All validation and model
// execution happen before the pool is changed, so any returned error leaves its
// entries and LRU order unchanged.
func (p *LlamaPrefixPool) Prefill(ctx *backend.Context, cacheSalt string, prompt []int) (*tensor.Tensor, int, error) {
	if p == nil || p.model == nil || p.capacity <= 0 {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixPool.Prefill needs an initialized pool")
	}
	if ctx == nil {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixPool.Prefill needs a context")
	}
	m := p.model
	if len(prompt) == 0 {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixPool.Prefill needs a non-empty prompt")
	}
	if len(prompt) > m.Config.Ctx {
		return nil, 0, fmt.Errorf("nlp: LlamaPrefixPool prompt length %d exceeds context %d", len(prompt), m.Config.Ctx)
	}
	for _, token := range prompt {
		if token < 0 || token >= m.Config.Vocab {
			return nil, 0, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
		}
	}
	if err := p.validatePool(); err != nil {
		return nil, 0, err
	}

	best, lcp, bestExact := p.match(cacheSalt, prompt)
	if best >= 0 {
		if err := p.validateSlot(best); err != nil {
			return nil, 0, err
		}
	}
	if bestExact {
		p.touch(best)
		last := p.slots[best].last
		return last.Cast(last.Dtype()), lcp, nil
	}

	keep, reused := lcp, lcp
	if best >= 0 && lcp == len(prompt) { // strict shortening
		keep = len(prompt) - 1
		reused = keep
	}
	var source *LlamaCache
	if best >= 0 {
		source = p.slots[best].cache
	}
	cache, err := refillLlamaCachePrefix(m, p.spare, source, keep)
	if err != nil {
		return nil, 0, err
	}
	logits, err := m.PrefillAppend(ctx, cache, prompt[keep:])
	if err != nil {
		return nil, 0, err
	}
	last, err := logits.Slice(0, logits.Shape()[0]-1, logits.Shape()[0])
	if err != nil {
		return nil, 0, err
	}
	p.spare = p.insert(llamaPrefixPoolSlot{
		salt: cacheSalt, tokens: append([]int(nil), prompt...),
		cache: cache, last: last.Cast(last.Dtype()), prev: -1, next: -1,
	})
	return last, reused, nil
}

func (p *LlamaPrefixPool) validatePool() error {
	if len(p.slots) > p.capacity {
		return fmt.Errorf("nlp: LlamaPrefixPool has %d entries beyond capacity %d", len(p.slots), p.capacity)
	}
	if len(p.slots) == 0 && (p.lru != -1 || p.mru != -1) {
		return fmt.Errorf("nlp: empty LlamaPrefixPool has LRU endpoints %d/%d", p.lru, p.mru)
	}
	if len(p.slots) > 0 && (p.lru < 0 || p.lru >= len(p.slots) || p.mru < 0 || p.mru >= len(p.slots)) {
		return fmt.Errorf("nlp: LlamaPrefixPool has invalid LRU endpoints %d/%d", p.lru, p.mru)
	}
	return nil
}

func (p *LlamaPrefixPool) validateSlot(i int) error {
	if i < 0 || i >= len(p.slots) {
		return fmt.Errorf("nlp: LlamaPrefixPool selected invalid entry %d", i)
	}
	s := &p.slots[i]
	n, err := validateLlamaPrefixState(p.model, s.cache)
	if err != nil {
		return fmt.Errorf("nlp: LlamaPrefixPool entry %d: %w", i, err)
	}
	if n == 0 || n != len(s.tokens) {
		return fmt.Errorf("nlp: LlamaPrefixPool entry %d has %d tokens and %d KV rows", i, len(s.tokens), n)
	}
	if s.last == nil || s.last.Ndim() != 2 || s.last.Shape()[0] != 1 || s.last.Shape()[1] != p.model.Config.Vocab {
		return fmt.Errorf("nlp: LlamaPrefixPool entry %d has invalid stored logits", i)
	}
	return nil
}

func (p *LlamaPrefixPool) touch(i int) {
	p.clock++
	p.slots[i].used = p.clock
	p.touchRadix(i)
	p.moveToMRU(i)
}

func (p *LlamaPrefixPool) insert(slot llamaPrefixPoolSlot) *LlamaCache {
	p.clock++
	slot.used = p.clock
	if len(p.slots) < p.capacity {
		i := len(p.slots)
		p.slots = append(p.slots, slot)
		p.linkMRU(i)
		p.addRadix(i)
		return nil
	}
	lru := p.lru
	evicted := p.slots[lru].cache
	p.removeRadix(lru)
	p.unlinkLRU(lru)
	p.slots[lru] = slot
	p.linkMRU(lru)
	p.addRadix(lru)
	return evicted
}

// refillLlamaCachePrefix rewinds dst while retaining its row-buffer allocations,
// then copies src[:n]. A pool keeps the last evicted cache as dst, so misses after
// warm-up ping-pong two backing allocations instead of allocating another full KV.
func refillLlamaCachePrefix(m *Llama, dst, src *LlamaCache, n int) (*LlamaCache, error) {
	if n < 0 {
		return nil, fmt.Errorf("nlp: cannot clone %d LlamaCache rows", n)
	}
	if n > 0 {
		if src == nil {
			return nil, fmt.Errorf("nlp: cannot clone %d rows from a nil LlamaCache", n)
		}
		rows, err := validateLlamaPrefixState(m, src)
		if err != nil {
			return nil, err
		}
		if n > rows {
			return nil, fmt.Errorf("nlp: cannot clone %d rows from a %d-row LlamaCache", n, rows)
		}
	}
	if dst == nil || len(dst.K) != len(m.Blocks) || len(dst.V) != len(m.Blocks) {
		dst = m.NewCache()
	} else if err := dst.truncate(0); err != nil {
		// A failed backend may leave the invisible scratch cache partially built.
		// Fall back to a fresh cache; visible entries and LRU state remain untouched.
		dst = m.NewCache()
	}
	dst.pos.reset()
	if n == 0 {
		return dst, nil
	}
	dst.bufs.k = growSlice(dst.bufs.k, len(m.Blocks))
	dst.bufs.v = growSlice(dst.bufs.v, len(m.Blocks))
	for l := range m.Blocks {
		dst.K[l] = dst.bufs.k[l].CopyPrefix(src.K[l], n)
		dst.V[l] = dst.bufs.v[l].CopyPrefix(src.V[l], n)
	}
	return dst, nil
}
