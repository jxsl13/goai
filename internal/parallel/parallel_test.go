package parallel

import (
	"sync"
	"sync/atomic"
	"testing"
)

// Every index must be visited exactly once, at every n around the worker count — an
// off-by-one in the chunking would either drop the tail or double-run it.
func TestRowsCoversEveryIndexExactlyOnce(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, 1000} {
		hits := make([]int32, n)
		Rows(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				atomic.AddInt32(&hits[i], 1)
			}
		})
		for i, h := range hits {
			if h != 1 {
				t.Fatalf("n=%d: index %d visited %d times, want 1", n, i, h)
			}
		}
	}
}

// Ranges must be contiguous, disjoint and ascending — the contract callers rely on to
// write output rows without synchronizing.
func TestRowsHandsOutDisjointContiguousRanges(t *testing.T) {
	const n = 500
	var mu sync.Mutex
	var got [][2]int
	Rows(n, func(lo, hi int) {
		mu.Lock()
		got = append(got, [2]int{lo, hi})
		mu.Unlock()
	})
	total := 0
	for _, r := range got {
		if r[0] >= r[1] {
			t.Fatalf("empty or inverted range %v", r)
		}
		total += r[1] - r[0]
	}
	if total != n {
		t.Fatalf("ranges cover %d indices, want %d", total, n)
	}
}

// THE NESTING GUARD, which is the reason this pool exists rather than a bare WaitGroup.
// A Rows call from inside a Rows body must still complete: submission is non-blocking,
// so an occupied mailbox runs the chunk inline instead of waiting for one to free. If
// this ever deadlocks the test times out, which is the failure mode being guarded.
func TestRowsNestedDoesNotDeadlock(t *testing.T) {
	const outer, inner = 64, 64
	var count atomic.Int64
	Rows(outer, func(lo, hi int) {
		for range hi - lo {
			Rows(inner, func(ilo, ihi int) {
				count.Add(int64(ihi - ilo))
			})
		}
	})
	if got, want := count.Load(), int64(outer*inner); got != want {
		t.Fatalf("nested Rows ran %d inner indices, want %d", got, want)
	}
}

// Three levels deep, because a guard that only handles one level of nesting would pass
// the test above and still deadlock in production under a concurrent caller.
func TestRowsDeeplyNestedDoesNotDeadlock(t *testing.T) {
	var count atomic.Int64
	Rows(8, func(a, b int) {
		Rows(8, func(c, d int) {
			Rows(8, func(e, f int) {
				count.Add(int64(f - e))
			})
		})
	})
	if count.Load() == 0 {
		t.Fatal("deeply nested Rows never ran its body")
	}
}

// Many concurrent callers must not deadlock either — the mailboxes are shared, so this
// is the same guard reached by contention rather than by nesting.
func TestRowsConcurrentCallersDoNotDeadlock(t *testing.T) {
	var wg sync.WaitGroup
	var count atomic.Int64
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			Rows(256, func(lo, hi int) { count.Add(int64(hi - lo)) })
		}()
	}
	wg.Wait()
	if got, want := count.Load(), int64(32*256); got != want {
		t.Fatalf("concurrent Rows covered %d indices, want %d", got, want)
	}
}

// RowsIdx must cover every index exactly once, exactly as Rows does, and hand out chunk
// indices inside [0, Workers()) so a caller can size a per-chunk buffer array by it.
func TestRowsIdxCoversEveryIndexAndBoundsTheChunkIndex(t *testing.T) {
	for _, n := range []int{1, 2, 7, 16, 17, 1000} {
		hits := make([]int32, n)
		var maxChunk atomic.Int32
		RowsIdx(n, func(chunk, lo, hi int) {
			if chunk < 0 || chunk >= Workers() {
				t.Errorf("n=%d: chunk index %d outside [0,%d)", n, chunk, Workers())
			}
			for {
				old := maxChunk.Load()
				if int32(chunk) <= old || maxChunk.CompareAndSwap(old, int32(chunk)) {
					break
				}
			}
			for i := lo; i < hi; i++ {
				atomic.AddInt32(&hits[i], 1)
			}
		})
		for i, h := range hits {
			if h != 1 {
				t.Fatalf("n=%d: index %d visited %d times, want 1", n, i, h)
			}
		}
	}
}

// Distinct chunks must get DISTINCT indices — a per-chunk buffer array is only safe if no
// two concurrent chunks share a slot.
func TestRowsIdxGivesDistinctChunkIndices(t *testing.T) {
	const n = 4096
	seen := make([]int32, Workers())
	RowsIdx(n, func(chunk, lo, hi int) { atomic.AddInt32(&seen[chunk], 1) })
	for c, k := range seen {
		if k > 1 {
			t.Fatalf("chunk index %d handed out %d times — per-chunk buffers would alias", c, k)
		}
	}
}
