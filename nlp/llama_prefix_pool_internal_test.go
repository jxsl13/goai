package nlp

import (
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type prefixPoolFailBackend struct {
	delegate backend.Backend
	calls    int
	failAt   int
}

func (b *prefixPoolFailBackend) Name() backend.Name    { return "prefix-pool-fail" }
func (b *prefixPoolFailBackend) Device() tensor.Device { return b.delegate.Device() }
func (b *prefixPoolFailBackend) Synchronize() error    { return b.delegate.Synchronize() }
func (b *prefixPoolFailBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	k, ok := b.delegate.Kernel(op, dt)
	if !ok {
		return nil, false
	}
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		b.calls++
		if b.calls == b.failAt {
			return nil, errors.New("injected prefix-pool backend failure")
		}
		return k(ctx.WithBackend(b.delegate), in, attrs)
	}, true
}

func findLlamaPrefixPoolSlot(p *LlamaPrefixPool, salt string, tokens []int) *llamaPrefixPoolSlot {
	for i := range p.slots {
		if p.slots[i].salt == salt && slices.Equal(p.slots[i].tokens, tokens) {
			return &p.slots[i]
		}
	}
	return nil
}

func assertLlamaPrefixPoolResult(t *testing.T, m *Llama, p *LlamaPrefixPool, salt string, prompt []int, got *tensor.Tensor) {
	t.Helper()
	wantCache := m.NewCache()
	full, err := m.Prefill(backend.NewContext(), wantCache, prompt)
	if err != nil {
		t.Fatal(err)
	}
	want, err := full.Slice(0, len(prompt)-1, len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	layerSkipTensorExact(t, got, want)
	slot := findLlamaPrefixPoolSlot(p, salt, prompt)
	if slot == nil {
		t.Fatalf("missing slot salt=%q prompt=%v", salt, prompt)
	}
	layerSkipCacheExact(t, slot.cache, wantCache)
	layerSkipTensorExact(t, slot.last, want)
}

// §T875/§R267: an intervening prompt must not destroy an older slot. Each
// return and retained slot is pinned to an independent full prefill.
func TestLlamaPrefixPoolSequenceExact(t *testing.T) {
	for _, tc := range []struct {
		name  string
		hooks bool
	}{
		{"gqa", false},
		{"qwen_granite_hooks_and_layer_backends", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := layerSkipTinyLlama(t, tc.hooks)
			pool, err := NewLlamaPrefixPool(m, 4)
			if err != nil {
				t.Fatal(err)
			}
			a := []int{1, 2, 3, 4}
			b := []int{1, 2, 5, 6}
			short := []int{1, 2}

			got, reused, err := pool.Prefill(backend.NewContext(), "team", a)
			if err != nil || reused != 0 {
				t.Fatalf("first: reused=%d err=%v", reused, err)
			}
			assertLlamaPrefixPoolResult(t, m, pool, "team", a, got)
			aSlot := findLlamaPrefixPoolSlot(pool, "team", a)
			aCache := aSlot.cache
			aK := aCache.K[0].Cast(aCache.K[0].Dtype())
			aV := aCache.V[0].Cast(aCache.V[0].Dtype())

			got, reused, err = pool.Prefill(backend.NewContext(), "team", b)
			if err != nil || reused != 2 {
				t.Fatalf("intervening: reused=%d err=%v", reused, err)
			}
			assertLlamaPrefixPoolResult(t, m, pool, "team", b, got)
			if aSlot = findLlamaPrefixPoolSlot(pool, "team", a); aSlot == nil || aSlot.cache != aCache {
				t.Fatal("source entry was replaced while cloning its prefix")
			}
			layerSkipTensorExact(t, aSlot.cache.K[0], aK)
			layerSkipTensorExact(t, aSlot.cache.V[0], aV)

			got, reused, err = pool.Prefill(backend.NewContext(), "team", a)
			if err != nil || reused != len(a) {
				t.Fatalf("return to A: reused=%d err=%v", reused, err)
			}
			assertLlamaPrefixPoolResult(t, m, pool, "team", a, got)

			got, reused, err = pool.Prefill(backend.NewContext(), "team", short)
			if err != nil || reused != len(short)-1 {
				t.Fatalf("shorten: reused=%d err=%v", reused, err)
			}
			assertLlamaPrefixPoolResult(t, m, pool, "team", short, got)

			// Exact older entry wins over the newer longer A/B entries even though all
			// have the same full LCP with short. Returned tensors are protected copies.
			got.SetF64(got.AtF64(0, 0)+1234, 0, 0)
			if _, reused, err = pool.Prefill(backend.NewContext(), "team", a); err != nil || reused != len(a) {
				t.Fatalf("make longer entry newer: reused=%d err=%v", reused, err)
			}
			got, reused, err = pool.Prefill(backend.NewContext(), "team", short)
			if err != nil || reused != len(short) {
				t.Fatalf("short exact: reused=%d err=%v", reused, err)
			}
			assertLlamaPrefixPoolResult(t, m, pool, "team", short, got)
		})
	}
}

func TestLlamaPrefixPoolSaltIsolation(t *testing.T) {
	m := layerSkipTinyLlama(t, false)
	pool, err := NewLlamaPrefixPool(m, 3)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 2, 3, 4}
	if _, reused, err := pool.Prefill(backend.NewContext(), "tenant-a", prompt); err != nil || reused != 0 {
		t.Fatalf("tenant-a first: reused=%d err=%v", reused, err)
	}
	got, reused, err := pool.Prefill(backend.NewContext(), "tenant-b", prompt)
	if err != nil || reused != 0 {
		t.Fatalf("cross-salt request reused %d, err=%v", reused, err)
	}
	assertLlamaPrefixPoolResult(t, m, pool, "tenant-b", prompt, got)
	got, reused, err = pool.Prefill(backend.NewContext(), "tenant-a", prompt)
	if err != nil || reused != len(prompt) {
		t.Fatalf("same-salt exact reused %d, err=%v", reused, err)
	}
	assertLlamaPrefixPoolResult(t, m, pool, "tenant-a", prompt, got)
	if pool.Len() != 2 {
		t.Fatalf("pool len %d, want 2 isolated entries", pool.Len())
	}
}

func TestLlamaPrefixPoolLRUEviction(t *testing.T) {
	m := layerSkipTinyLlama(t, false)
	pool, err := NewLlamaPrefixPool(m, 2)
	if err != nil {
		t.Fatal(err)
	}
	a, b, c := []int{0, 1}, []int{2, 3}, []int{4, 5}
	for _, prompt := range [][]int{a, b} {
		if _, reused, err := pool.Prefill(backend.NewContext(), "", prompt); err != nil || reused != 0 {
			t.Fatalf("insert %v: reused=%d err=%v", prompt, reused, err)
		}
	}
	if _, reused, err := pool.Prefill(backend.NewContext(), "", a); err != nil || reused != len(a) {
		t.Fatalf("touch A: reused=%d err=%v", reused, err)
	}
	if _, reused, err := pool.Prefill(backend.NewContext(), "", c); err != nil || reused != 0 {
		t.Fatalf("insert C: reused=%d err=%v", reused, err)
	}
	if findLlamaPrefixPoolSlot(pool, "", a) == nil || findLlamaPrefixPoolSlot(pool, "", c) == nil {
		t.Fatal("MRU entries A/C missing")
	}
	if findLlamaPrefixPoolSlot(pool, "", b) != nil {
		t.Fatal("least-recently-used B was not evicted")
	}
}

func TestLlamaPrefixPoolRecyclesEvictedKVBacking(t *testing.T) {
	m := layerSkipTinyLlama(t, false)
	pool, err := NewLlamaPrefixPool(m, 1)
	if err != nil {
		t.Fatal(err)
	}
	a := []int{1, 2, 3, 4}
	b := []int{1, 2, 5, 6}
	if _, _, err := pool.Prefill(backend.NewContext(), "team", a); err != nil {
		t.Fatal(err)
	}
	aCache := pool.slots[0].cache
	aKStorage, aVStorage := aCache.K[0].Storage(), aCache.V[0].Storage()
	if _, reused, err := pool.Prefill(backend.NewContext(), "team", b); err != nil || reused != 2 {
		t.Fatalf("B: reused=%d err=%v", reused, err)
	}
	bCache := pool.slots[0].cache
	if pool.spare != aCache {
		t.Fatal("evicted A cache did not become the spare")
	}
	got, reused, err := pool.Prefill(backend.NewContext(), "team", a)
	if err != nil || reused != 2 {
		t.Fatalf("A again: reused=%d err=%v", reused, err)
	}
	aAgain := pool.slots[0].cache
	if aAgain != aCache || aAgain.K[0].Storage() != aKStorage || aAgain.V[0].Storage() != aVStorage {
		t.Fatal("A did not reuse its original cache and row-buffer storage")
	}
	if pool.spare != bCache {
		t.Fatal("evicted B cache did not become the next spare")
	}
	assertLlamaPrefixPoolResult(t, m, pool, "team", a, got)
}

func TestRowBufCopyPrefixRetainsStorageAndCopiesValues(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32, tensor.F16, tensor.BF16} {
		t.Run(dt.String(), func(t *testing.T) {
			src := tensor.New(dt, tensor.Shape{6, 3})
			for i := range src.Numel() {
				idx := tensor.Unravel(i, src.Shape())
				src.SetF64(float64(i)/7-1, idx...)
			}
			snapshot := src.Cast(dt)
			old := tensor.New(dt, tensor.Shape{4, 3})
			var rb rowBuf
			cur := rb.Append(nil, old)
			storage := cur.Storage()
			cur = rb.Truncate(cur, 0)
			got := rb.CopyPrefix(src, 3)
			if got.Storage() != storage {
				t.Fatal("compatible retained storage was replaced")
			}
			want, err := src.Slice(0, 0, 3)
			if err != nil {
				t.Fatal(err)
			}
			layerSkipTensorExact(t, got, want)
			got.SetF64(got.AtF64(0, 0)+1, 0, 0)
			layerSkipTensorExact(t, src, snapshot)
		})
	}
}

func TestLlamaPrefixPoolBackendErrorIsTransactional(t *testing.T) {
	m := layerSkipTinyLlama(t, false)
	pool, err := NewLlamaPrefixPool(m, 1)
	if err != nil {
		t.Fatal(err)
	}
	a, b, c := []int{1, 2, 3}, []int{1, 2, 4}, []int{1, 2, 5}
	if _, _, err := pool.Prefill(backend.NewContext(), "team", a); err != nil {
		t.Fatal(err)
	}
	if _, _, err := pool.Prefill(backend.NewContext(), "team", b); err != nil {
		t.Fatal(err)
	}
	assertLlamaPrefixPoolPolicyStructure(t, pool)
	wantSlot, wantLCP, wantExact := pool.match("team", c)
	beforeClock := pool.clock
	before := pool.slots[0]
	spare := pool.spare
	fail := &prefixPoolFailBackend{delegate: backend.Reference(), failAt: 2}
	if _, _, err := pool.Prefill(&backend.Context{Backend: fail}, "team", c); err == nil {
		t.Fatal("injected backend failure was not returned")
	}
	if pool.clock != beforeClock || len(pool.slots) != 1 || pool.slots[0].cache != before.cache ||
		pool.slots[0].last != before.last || pool.slots[0].used != before.used ||
		!slices.Equal(pool.slots[0].tokens, before.tokens) {
		t.Fatal("backend failure mutated visible entries or LRU state")
	}
	if pool.spare != spare {
		t.Fatal("backend failure replaced the reusable scratch allocation")
	}
	if gotSlot, gotLCP, gotExact := pool.match("team", c); gotSlot != wantSlot || gotLCP != wantLCP || gotExact != wantExact {
		t.Fatal("backend failure mutated the radix lookup result")
	}
	assertLlamaPrefixPoolPolicyStructure(t, pool)
	got, reused, err := pool.Prefill(backend.NewContext(), "team", c)
	if err != nil || reused != 2 {
		t.Fatalf("recovery: reused=%d err=%v", reused, err)
	}
	assertLlamaPrefixPoolResult(t, m, pool, "team", c, got)
	assertLlamaPrefixPoolPolicyStructure(t, pool)
}

func TestLlamaPrefixPoolValidationDoesNotMutateState(t *testing.T) {
	if _, err := NewLlamaPrefixPool(nil, 1); err == nil {
		t.Fatal("nil model accepted")
	}
	m := layerSkipTinyLlama(t, true)
	if _, err := NewLlamaPrefixPool(m, 0); err == nil {
		t.Fatal("zero capacity accepted")
	}
	if _, _, err := (*LlamaPrefixPool)(nil).Prefill(backend.NewContext(), "", []int{1}); err == nil {
		t.Fatal("nil pool accepted")
	}
	pool, err := NewLlamaPrefixPool(m, 2)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 2, 3}
	if _, _, err := pool.Prefill(backend.NewContext(), "team", prompt); err != nil {
		t.Fatal(err)
	}
	beforeClock := pool.clock
	before := pool.slots[0]
	beforeTokens := append([]int(nil), before.tokens...)

	assertUnchanged := func(t *testing.T) {
		t.Helper()
		if pool.clock != beforeClock || len(pool.slots) != 1 {
			t.Fatalf("LRU state mutated: clock=%d slots=%d", pool.clock, len(pool.slots))
		}
		got := pool.slots[0]
		if got.cache != before.cache || got.last != before.last || got.used != before.used || got.salt != before.salt || !slices.Equal(got.tokens, beforeTokens) {
			t.Fatal("entry metadata mutated after validation error")
		}
	}
	tooLong := make([]int, m.Config.Ctx+1)
	for _, tc := range []struct {
		name   string
		ctx    *backend.Context
		prompt []int
	}{
		{"nil_context", nil, prompt},
		{"empty", backend.NewContext(), nil},
		{"negative_token", backend.NewContext(), []int{1, -1}},
		{"high_token", backend.NewContext(), []int{1, m.Config.Vocab}},
		{"over_context", backend.NewContext(), tooLong},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := pool.Prefill(tc.ctx, "team", tc.prompt); err == nil {
				t.Fatal("invalid request accepted")
			}
			assertUnchanged(t)
		})
	}
	m.LayerBackends = m.LayerBackends[:1]
	if _, _, err := pool.Prefill(backend.NewContext(), "team", []int{1, 2, 4}); err == nil {
		t.Fatal("stale layer plan accepted")
	}
	m.LayerBackends = []backend.Name{backend.Ref, backend.Ref}
	assertUnchanged(t)
}

func linearLlamaPrefixPoolMatch(slots []llamaPrefixPoolSlot, salt string, prompt []int) (best, lcp int, exact bool) {
	best, lcp = -1, 0
	for i := range slots {
		s := &slots[i]
		if s.salt != salt {
			continue
		}
		n := commonTokenPrefix(s.tokens, prompt)
		isExact := n == len(prompt) && len(s.tokens) == len(prompt)
		if best < 0 || n > lcp || n == lcp && (isExact && !exact || isExact == exact && s.used > slots[best].used) {
			best, lcp, exact = i, n, isExact
		}
	}
	return best, lcp, exact
}

func assertLlamaPrefixPoolPolicyStructure(t testing.TB, p *LlamaPrefixPool) {
	t.Helper()
	seenLRU := make([]bool, len(p.slots))
	prev, count := -1, 0
	var lastUsed uint64
	for i := p.lru; i >= 0; i = p.slots[i].next {
		if i >= len(p.slots) || seenLRU[i] {
			t.Fatalf("invalid/cyclic LRU slot %d", i)
		}
		seenLRU[i] = true
		if p.slots[i].prev != prev {
			t.Fatalf("LRU slot %d prev=%d want %d", i, p.slots[i].prev, prev)
		}
		if count > 0 && p.slots[i].used <= lastUsed {
			t.Fatalf("LRU use order %d after %d", p.slots[i].used, lastUsed)
		}
		lastUsed, prev, count = p.slots[i].used, i, count+1
	}
	if count != len(p.slots) || count > 0 && prev != p.mru || count == 0 && (p.lru != -1 || p.mru != -1) {
		t.Fatalf("LRU coverage/endpoints count=%d slots=%d tail=%d endpoints=%d/%d", count, len(p.slots), prev, p.lru, p.mru)
	}

	seenRadix := make([]bool, len(p.slots))
	for salt, root := range p.roots {
		if len(root.edge) != 0 || root.terminal != -1 {
			t.Fatalf("salt %q has malformed radix root", salt)
		}
		var walk func(*llamaPrefixRadixNode, []int, bool) int
		walk = func(node *llamaPrefixRadixNode, prefix []int, isRoot bool) int {
			if !isRoot && len(node.edge) == 0 {
				t.Fatal("empty non-root radix edge")
			}
			if !isRoot && node.terminal < 0 && len(node.children) == 1 {
				t.Fatalf("uncompressed unary radix node at %v", prefix)
			}
			path := append(append([]int(nil), prefix...), node.edge...)
			best := node.terminal
			if node.terminal >= 0 {
				i := node.terminal
				if i >= len(p.slots) || seenRadix[i] || p.slots[i].salt != salt || !slices.Equal(path, p.slots[i].tokens) {
					t.Fatalf("bad radix terminal salt=%q slot=%d path=%v", salt, i, path)
				}
				seenRadix[i] = true
			}
			for first, child := range node.children {
				if len(child.edge) == 0 || child.edge[0] != first {
					t.Fatalf("radix child key=%d edge=%v", first, child.edge)
				}
				candidate := walk(child, path, false)
				if p.newerSlot(candidate, best) {
					best = candidate
				}
			}
			if node.best != best {
				t.Fatalf("radix best=%d want %d at %v", node.best, best, path)
			}
			return best
		}
		walk(root, nil, true)
	}
	for i := range p.slots {
		if !seenLRU[i] || !seenRadix[i] {
			t.Fatalf("slot %d missing: lru=%v radix=%v", i, seenLRU[i], seenRadix[i])
		}
	}
}

func TestLlamaPrefixPoolRadixMatchesLinearPolicy(t *testing.T) {
	const capacity = 31
	p := &LlamaPrefixPool{
		capacity: capacity, roots: make(map[string]*llamaPrefixRadixNode), lru: -1, mru: -1,
	}
	rng := rand.New(rand.NewSource(878))
	salts := []string{"", "tenant-a", "tenant-b"}
	for step := range 1200 {
		if len(p.slots) > 0 && step%3 == 0 {
			p.touch(rng.Intn(len(p.slots)))
		} else {
			saltID := rng.Intn(len(salts))
			n := 4 + rng.Intn(20)
			tokens := make([]int, n)
			for i := range tokens {
				if i < 3 {
					tokens[i] = saltID*100 + i
				} else {
					tokens[i] = (step/(i+1) + i*17) % 53
				}
			}
			tokens[n-1] = 10000 + step // unique terminal within this run
			p.insert(llamaPrefixPoolSlot{salt: salts[saltID], tokens: tokens, prev: -1, next: -1})
		}
		assertLlamaPrefixPoolPolicyStructure(t, p)

		for q := range 12 {
			salt := salts[rng.Intn(len(salts))]
			var prompt []int
			if len(p.slots) > 0 && q < 9 {
				s := &p.slots[rng.Intn(len(p.slots))]
				salt = s.salt
				switch q % 3 {
				case 0: // exact
					prompt = append([]int(nil), s.tokens...)
				case 1: // strict shortening, often inside a compressed edge
					prompt = append([]int(nil), s.tokens[:1+rng.Intn(len(s.tokens)-1)]...)
				case 2: // divergence after a real prefix
					prompt = append([]int(nil), s.tokens...)
					at := rng.Intn(len(prompt))
					prompt[at] += 100000
				}
			} else {
				prompt = []int{rng.Intn(400), rng.Intn(400), rng.Intn(400)}
			}
			wantSlot, wantLCP, wantExact := linearLlamaPrefixPoolMatch(p.slots, salt, prompt)
			gotSlot, gotLCP, gotExact := p.match(salt, prompt)
			if gotSlot != wantSlot || gotLCP != wantLCP || gotExact != wantExact {
				t.Fatalf("step=%d q=%d salt=%q prompt=%v: indexed=(%d,%d,%v) linear=(%d,%d,%v)",
					step, q, salt, prompt, gotSlot, gotLCP, gotExact, wantSlot, wantLCP, wantExact)
			}
		}
	}
}

var llamaPrefixPoolBenchOut *tensor.Tensor
var llamaPrefixCopyBenchOut *LlamaCache
var llamaPrefixMatchBenchOut int

// Isolates T877's one changed operation at realistic layer count. The reference
// closure is T876's exact validate→truncate→Slice→Append implementation.
func BenchmarkLlamaCachePrefixCopy(b *testing.B) {
	m, err := NewLlama(LlamaConfig{
		Vocab: 128, Ctx: 128, Dim: 16, Heads: 4, KVHeads: 2,
		Layers: 32, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 17)
	if err != nil {
		b.Fatal(err)
	}
	const prefix = 96
	tokens := make([]int, prefix)
	for i := range tokens {
		tokens[i] = (i*17 + 3) % m.Config.Vocab
	}
	src := m.NewCache()
	if _, err := m.Prefill(&backend.Context{Backend: backend.Reference()}, src, tokens); err != nil {
		b.Fatal(err)
	}

	b.Run("t876_slice_append", func(b *testing.B) {
		dst := m.NewCache()
		b.ReportAllocs()
		for range b.N {
			if _, err := validateLlamaPrefixState(m, src); err != nil {
				b.Fatal(err)
			}
			if err := dst.truncate(0); err != nil {
				b.Fatal(err)
			}
			dst.pos.reset()
			for l := range m.Blocks {
				k, _ := src.K[l].Slice(0, 0, prefix)
				v, _ := src.V[l].Slice(0, 0, prefix)
				dst.K[l], dst.V[l] = dst.bufs.appendKV(dst.K, dst.V, l, k, v)
			}
			llamaPrefixCopyBenchOut = dst
		}
	})

	b.Run("t877_direct_copy", func(b *testing.B) {
		dst := m.NewCache()
		b.ReportAllocs()
		for range b.N {
			var err error
			dst, err = refillLlamaCachePrefix(m, dst, src, prefix)
			if err != nil {
				b.Fatal(err)
			}
			llamaPrefixCopyBenchOut = dst
		}
	})
}

func BenchmarkLlamaPrefixPoolLookup(b *testing.B) {
	const slots, tokens = 1024, 256
	p := &LlamaPrefixPool{
		capacity: slots, roots: make(map[string]*llamaPrefixRadixNode), lru: -1, mru: -1,
	}
	for i := range slots {
		prompt := make([]int, tokens)
		for j := range prompt {
			if j < 192 {
				prompt[j] = (j*17 + 3) % 4096
			} else {
				prompt[j] = 4096 + i*64 + j
			}
		}
		p.insert(llamaPrefixPoolSlot{salt: "team", tokens: prompt, prev: -1, next: -1})
	}
	assertLlamaPrefixPoolPolicyStructure(b, p)

	b.Run("linear_scan_touch", func(b *testing.B) {
		linear := append([]llamaPrefixPoolSlot(nil), p.slots...)
		clock := p.clock
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			prompt := linear[i&(slots-1)].tokens
			best, lcp, exact := linearLlamaPrefixPoolMatch(linear, "team", prompt)
			if !exact || lcp != tokens {
				b.Fatal("linear lookup lost exact prompt")
			}
			clock++
			linear[best].used = clock
			llamaPrefixMatchBenchOut = best
		}
	})

	b.Run("compressed_radix_touch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			prompt := p.slots[i&(slots-1)].tokens
			best, lcp, exact := p.match("team", prompt)
			if !exact || lcp != tokens {
				b.Fatal("radix lookup lost exact prompt")
			}
			p.touch(best)
			llamaPrefixMatchBenchOut = best
		}
	})

	if len(p.slots) != slots {
		b.Fatal(fmt.Errorf("lookup benchmark retained %d slots", len(p.slots)))
	}
}

func BenchmarkLlamaPrefixPoolRepeatedPrompts(b *testing.B) {
	m, err := NewLlama(LlamaConfig{
		Vocab: 128, Ctx: 128, Dim: 16, Heads: 4, KVHeads: 2,
		Layers: 3, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 17)
	if err != nil {
		b.Fatal(err)
	}
	const prefixLen, suffixLen = 96, 8
	prompts := [2][]int{make([]int, prefixLen+suffixLen), make([]int, prefixLen+suffixLen)}
	for i := range prefixLen {
		tok := (i*17 + 3) % m.Config.Vocab
		prompts[0][i], prompts[1][i] = tok, tok
	}
	for i := range suffixLen {
		prompts[0][prefixLen+i] = (i*11 + 5) % m.Config.Vocab
		prompts[1][prefixLen+i] = (i*13 + 9) % m.Config.Vocab
	}
	ctx := &backend.Context{Backend: backend.Reference()}

	b.Run("full_recompute", func(b *testing.B) {
		b.ReportAllocs()
		for i := range b.N {
			out, err := m.Prefill(ctx, m.NewCache(), prompts[i&1])
			if err != nil {
				b.Fatal(err)
			}
			llamaPrefixPoolBenchOut = out
		}
	})

	b.Run("single_slot_reuse_96", func(b *testing.B) {
		cache, err := NewLlamaPrefixCache(m)
		if err != nil {
			b.Fatal(err)
		}
		if _, _, err := cache.Prefill(ctx, prompts[0]); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			out, reused, err := cache.Prefill(ctx, prompts[(i+1)&1])
			if err != nil || reused != prefixLen {
				b.Fatalf("single-slot reused=%d err=%v", reused, err)
			}
			llamaPrefixPoolBenchOut = out
		}
	})

	b.Run("one_pool_slot_clone_96", func(b *testing.B) {
		pool, err := NewLlamaPrefixPool(m, 1)
		if err != nil {
			b.Fatal(err)
		}
		if _, _, err := pool.Prefill(ctx, "team", prompts[0]); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			out, reused, err := pool.Prefill(ctx, "team", prompts[(i+1)&1])
			if err != nil || reused != prefixLen {
				b.Fatalf("one-slot pool reused=%d err=%v", reused, err)
			}
			llamaPrefixPoolBenchOut = out
		}
	})

	b.Run("two_slot_exact_reuse_104", func(b *testing.B) {
		pool, err := NewLlamaPrefixPool(m, 2)
		if err != nil {
			b.Fatal(err)
		}
		for i := range prompts {
			if _, _, err := pool.Prefill(ctx, "team", prompts[i]); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			out, reused, err := pool.Prefill(ctx, "team", prompts[i&1])
			if err != nil || reused != prefixLen+suffixLen {
				b.Fatalf("pool reused=%d err=%v", reused, err)
			}
			llamaPrefixPoolBenchOut = out
		}
	})
}
