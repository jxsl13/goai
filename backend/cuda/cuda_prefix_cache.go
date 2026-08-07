//go:build cuda && cgo && (linux || windows)

package cuda

// PrefixBlockCache is an automatic prefix cache over a PagedKVPool — the RadixAttention / vLLM automatic-
// prefix-caching policy layer on top of the refcounted block sharing (NewSeqSharingPrefix). It maps a
// content hash of each token block, CHAINED over all prior tokens (so a block matches only when the whole
// prefix does — vLLM's block-hashing scheme), to the physical KV block holding that prefix. On a new
// request Match returns the physical blocks of the longest cached WHOLE-block prefix, which the caller
// turns into a shared sequence (PagedKVPool.NewSeqFromBlocks) and prefills only the divergent suffix —
// so a shared prefix's KV is stored and computed ONCE across concurrent requests.
//
// The cache holds one reference on each cached block (so it survives after its owning sequence releases);
// entries are dropped FIFO when the count exceeds maxEntries, releasing the pool reference. Not
// synchronized — drive it from the single serving loop, like the pool.
type PrefixBlockCache struct {
	pool         *PagedKVPool
	blockSize    int
	byHash       map[uint64]int32 // chained-prefix block hash -> physical block id (cache holds 1 ref each)
	order        []uint64         // insertion order for FIFO eviction
	maxEntries   int
	Hits, Misses int // block-granular match stats
}

// NewPrefixBlockCache builds a cache over pool that pins at most maxEntries prefix blocks.
func NewPrefixBlockCache(pool *PagedKVPool, maxEntries int) *PrefixBlockCache {
	if maxEntries < 1 {
		maxEntries = 1
	}
	return &PrefixBlockCache{pool: pool, blockSize: pool.blockSize, byHash: make(map[uint64]int32), maxEntries: maxEntries}
}

const fnvOffset64 = 1469598103934665603
const fnvPrime64 = 1099511628211

// chainHash folds one token block into the running FNV-1a hash of all prior tokens.
func chainHash(prev uint64, toks []int32) uint64 {
	h := prev
	if h == 0 {
		h = fnvOffset64
	}
	for _, t := range toks {
		u := uint64(uint32(t))
		h ^= u & 0xff
		h *= fnvPrime64
		h ^= (u >> 8) & 0xff
		h *= fnvPrime64
		h ^= (u >> 16) & 0xff
		h *= fnvPrime64
		h ^= (u >> 24) & 0xff
		h *= fnvPrime64
	}
	return h
}

// Match returns the physical blocks of the longest cached WHOLE-block prefix of tokens (block-granular),
// updating Hits/Misses. The returned blocks are NOT increfed here — pass them to NewSeqFromBlocks (which
// increfs) to build the shared sequence.
func (c *PrefixBlockCache) Match(tokens []int32) []int32 {
	nb := len(tokens) / c.blockSize
	var out []int32
	var h uint64
	for i := 0; i < nb; i++ {
		h = chainHash(h, tokens[i*c.blockSize:(i+1)*c.blockSize])
		b, ok := c.byHash[h]
		if !ok {
			c.Misses++
			break
		}
		c.Hits++
		out = append(out, b)
	}
	return out
}

// Insert records blockIds under the chained hashes of tokens (pinning each block with a cache reference),
// skipping blocks already cached, and evicts FIFO when over maxEntries. tokens must cover at least
// len(blockIds)·blockSize tokens (blockIds[i] holds tokens[i*bs:(i+1)*bs]).
func (c *PrefixBlockCache) Insert(tokens []int32, blockIds []int32) {
	var h uint64
	for i := 0; i < len(blockIds); i++ {
		h = chainHash(h, tokens[i*c.blockSize:(i+1)*c.blockSize])
		if _, ok := c.byHash[h]; ok {
			continue // this prefix block already cached (already shared)
		}
		c.byHash[h] = blockIds[i]
		c.pool.incref(blockIds[i]) // cache pins the block
		c.order = append(c.order, h)
	}
	for len(c.byHash) > c.maxEntries && len(c.order) > 0 {
		ev := c.order[0]
		c.order = c.order[1:]
		if b, ok := c.byHash[ev]; ok {
			delete(c.byHash, ev)
			c.pool.release(b) // drop the cache reference
		}
	}
}

// Len reports how many prefix blocks are currently pinned.
func (c *PrefixBlockCache) Len() int { return len(c.byHash) }
