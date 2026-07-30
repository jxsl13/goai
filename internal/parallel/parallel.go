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
)

type task struct {
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
				if t.bodyIdx != nil {
					t.bodyIdx(t.chunk, t.lo, t.hi)
				} else {
					t.body(t.lo, t.hi)
				}
				t.wg.Done()
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
func RowsIdx(n int, body func(chunk, lo, hi int)) {
	parts := min(len(mailboxes)+1, n)
	if parts <= 1 {
		body(0, 0, n)
		return
	}
	chunk := (n + parts - 1) / parts
	var wg sync.WaitGroup
	var inline [][3]int
	mb, ci := 0, 1
	for lo := chunk; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		c := ci
		ci++
		queued := false
		for mb < len(mailboxes) && !queued {
			wg.Add(1)
			select {
			case mailboxes[mb] <- task{bodyIdx: body, chunk: c, lo: lo, hi: hi, wg: &wg}:
				queued = true
			default:
				wg.Done()
			}
			mb++
		}
		if !queued {
			inline = append(inline, [3]int{c, lo, hi})
		}
	}
	body(0, 0, min(chunk, n)) // the caller works chunk 0
	for _, r := range inline {
		body(r[0], r[1], r[2])
	}
	wg.Wait()
}

// Rows splits [0,n) into contiguous chunks and calls body on each, returning once every
// chunk has run. body must be safe to run concurrently for disjoint ranges: the caller
// owns that guarantee, and for a matmul it holds because each output row is written
// exactly once by exactly one chunk.
//
// The partition is deterministic but the ORDER is not, so body must not accumulate
// across chunks. Callers whose result must be bit-identical to the serial form need each
// chunk to compute the same values it would have computed alone — writing distinct
// indices, not reducing into a shared one.
func Rows(n int, body func(lo, hi int)) {
	parts := min(len(mailboxes)+1, n)
	if parts <= 1 {
		body(0, n)
		return
	}
	chunk := (n + parts - 1) / parts
	var wg sync.WaitGroup
	var inline [][2]int // ranges no worker accepted; run by the caller below
	mb := 0
	for lo := chunk; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		queued := false
		for mb < len(mailboxes) && !queued {
			wg.Add(1)
			select {
			case mailboxes[mb] <- task{body: body, lo: lo, hi: hi, wg: &wg}:
				queued = true
			default:
				wg.Done() // occupied: try the next worker, then fall back to inline
			}
			mb++
		}
		if !queued {
			inline = append(inline, [2]int{lo, hi})
		}
	}
	body(0, min(chunk, n)) // the caller works too, rather than idling on the barrier
	for _, r := range inline {
		body(r[0], r[1])
	}
	wg.Wait()
}
