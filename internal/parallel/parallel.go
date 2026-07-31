// Package parallel runs index-partitioned CPU work across a bounded worker pool.
//
// It exists because format/gguf's single-token matmul is 82% of a quantized decode step
// and its output rows are independent, but a package on the innermost path must not
// multiply the process's goroutine count every time a caller is already concurrent.
//
// The pool is fixed at GOMAXPROCS-1 workers and submission is NON-BLOCKING: when a
// worker's mailbox is occupied — a nested call from inside a worker, or simple
// contention — that chunk runs inline on the caller instead. Nesting therefore degrades
// to serial execution rather than fanning out again, and the pool cannot deadlock no
// matter how deep the nesting goes. This is the discipline backend/cpu arrived at; that
// package keeps its own pool, tuned with spin/steal/dense-regime machinery for the
// µs-scale barriers a decode step issues between kernels. Nothing here needs that: the
// chunks this pool hands out are tens of microseconds of dense arithmetic.
package parallel

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// job is one Rows/RowsIdx call's shared claim state. Workers pull units from next until it
// passes units; the slot they were given stays fixed for the whole drain, which is what a
// caller's per-slot scratch is keyed by.
type job struct {
	body    func(lo, hi int)
	bodyIdx func(slot, lo, hi int)
	next    atomic.Int64
	n       int
	units   int
	grain   int
	wg      sync.WaitGroup
}

// drain claims units until the cursor is exhausted. slot is this runner's stable index.
func (j *job) drain(slot int) {
	for {
		u := int(j.next.Add(1)) - 1
		if u >= j.units {
			return
		}
		lo := u * j.grain
		hi := min(lo+j.grain, j.n)
		if j.bodyIdx != nil {
			j.bodyIdx(slot, lo, hi)
		} else {
			j.body(lo, hi)
		}
	}
}

type task struct {
	job  *job
	slot int
	body func(lo, hi int)
	// bodyIdx and chunk serve RowsIdx. Carrying the chunk index in the struct rather than
	// capturing it in a per-chunk closure is what keeps RowsIdx allocation-free: measured at
	// 13 allocations and 312 B per call against Rows' 2 and 48 B, one closure for every chunk.
	// The struct is copied by memmove straight into the receiving worker's frame, so the two
	// extra words cost nothing next to one avoided heap allocation.
	bodyIdx func(chunk, lo, hi int)
	chunk   int
	lo, hi  int
	wg      *sync.WaitGroup
}

// grainPerWorker is how many units each worker is expected to claim on average. One would be a
// static split by another name, and more is not better: swept at 2, 4 and 8 against both axes,
// 2 was best on each. Dispatch cost rises monotonically with the grain (19.88, 23.00, 26.95 us on
// BenchmarkDispatchRowsIdx) while the balance win saturates immediately (127.1, 129.0, 128.9 ms on
// classic BenchmarkGBMHist_exact_20k). At 2 the dispatch path is FASTER than the static split it
// replaced (22.11 us), because the pool now receives one task per worker instead of one per chunk.
const grainPerWorker = 2

var mailboxes []chan task

func init() {
	// GOMAXPROCS-1, because the caller always works one chunk itself — so a full-width
	// split occupies every P exactly once with no spare thread to churn.
	n := max(runtime.GOMAXPROCS(0)-1, 0)
	mailboxes = make([]chan task, n)
	for i := range mailboxes {
		// UNBUFFERED, and that is the whole nesting guard. With a one-slot buffer a send
		// succeeds against a worker that is merely alive, including one blocked in its
		// own Rows barrier — which will not return to `range ch` until that barrier
		// clears, so the sender waits forever for a chunk nobody will run. Unbuffered, a
		// non-blocking send lands only on a worker parked in the receive right now, so
		// "the pool is busy" and "the send fails" are the same event and the fallback
		// below is always reachable. Buffering this channel reintroduces the deadlock.
		ch := make(chan task)
		mailboxes[i] = ch
		go func() {
			for t := range ch {
				t.job.drain(t.slot)
				t.job.wg.Done()
			}
		}()
	}
}

// Workers reports the maximum number of chunks Rows will split into, so callers can size
// a work threshold against the parallelism actually available.
func Workers() int { return len(mailboxes) + 1 }

// RowsIdx is Rows with the CHUNK INDEX passed to body, so a caller can keep one scratch
// buffer per chunk on its own struct instead of allocating one per call.
//
// This exists because the obvious spelling — allocating the buffer inside the body — is
// correct but can be ruinously expensive when the parallel call is itself in a loop. The
// GBM exact grower calls its split search and partition once PER TREE NODE; per-call
// scratch took that fit from 64MB and 883 allocs to 2007MB and 8965, a 31x memory
// regression hiding behind a 2.80x speedup. chunk is always in [0, Workers()).
func RowsIdx(n int, body func(slot, lo, hi int)) {
	run(&job{bodyIdx: body}, n)
}

// Rows splits [0,n) into contiguous units and calls body on each, returning once every unit
// has run. body must be safe to run concurrently for disjoint ranges: the caller owns that
// guarantee, and for a matmul it holds because each output row is written exactly once by
// exactly one unit.
//
// The partition is deterministic but the ORDER is not, so body must not accumulate across
// units. Callers whose result must be bit-identical to the serial form need each unit to
// compute the same values it would have computed alone — writing distinct indices, not
// reducing into a shared one.
func Rows(n int, body func(lo, hi int)) {
	run(&job{body: body}, n)
}

// run splits [0,n) into units and drains them across the pool plus the calling goroutine.
//
// UNITS ARE CLAIMED, NOT DEALT, and the grain is deliberately finer than one unit per worker.
// An equal static split assumes every worker retires its share at the same rate, which on a
// heterogeneous CPU is false: an M2 Pro has 8 performance and 4 efficiency cores, so the unit
// landing on an E core sets the barrier. The signature is that MORE CORES MAKE IT SLOWER —
// classic BenchmarkGBMHist_exact_20k measured 132.4ms at GOMAXPROCS=8 against 136.7ms at 12
// before this change (STATIC-CHUNKS-LOSE-ON-HETEROGENEOUS-CORES-001).
//
// grainPerWorker trades balance against claim overhead. The units this pool hands out are tens
// of microseconds of dense arithmetic, so an atomic increment per unit is negligible, but too
// fine a grain starts to matter and too coarse a one brings the tail back.
func run(j *job, n int) {
	parts := min(len(mailboxes)+1, n)
	if parts <= 1 {
		if j.bodyIdx != nil {
			j.bodyIdx(0, 0, n)
		} else {
			j.body(0, n)
		}
		return
	}
	units := min(parts*grainPerWorker, n)
	j.n, j.units = n, units
	j.grain = (n + units - 1) / units
	j.units = (n + j.grain - 1) / j.grain // recompute: the ceil grain may need fewer units

	// Slot 0 belongs to the calling goroutine, so a worker's slot is its mailbox index plus
	// one. Slots are STABLE for the whole drain and no two concurrent runners share one, which
	// is the property a per-slot scratch buffer actually needs — the old chunk index happened
	// to be unique per chunk, but with claiming a runner takes many units.
	for mb := 0; mb < len(mailboxes); mb++ {
		j.wg.Add(1)
		select {
		case mailboxes[mb] <- task{job: j, slot: mb + 1}:
		default:
			j.wg.Done() // occupied: this worker sits out; the cursor still gets drained
		}
	}
	j.drain(0) // the caller works too, rather than idling on the barrier
	j.wg.Wait()
}
