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
	body   func(lo, hi int)
	lo, hi int
	wg     *sync.WaitGroup
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
				t.body(t.lo, t.hi)
				t.wg.Done()
			}
		}()
	}
}

// Workers reports the maximum number of chunks Rows will split into, so callers can size
// a work threshold against the parallelism actually available.
func Workers() int { return len(mailboxes) + 1 }

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
			case mailboxes[mb] <- task{body, lo, hi, &wg}:
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

// SumF64 computes Σ chunkSum over a contiguous partition of [0,n): it splits the range exactly as
// [Rows] does, calls chunkSum(lo,hi) once per chunk (concurrently, on disjoint ranges), and returns
// the sum of the per-chunk partials in a FIXED chunk order. The partition and the final combine order
// are both deterministic, so the result is reproducible run-to-run — but it is a REASSOCIATION of the
// serial left-to-right sum, so callers must tolerate the ~1-ULP-per-level rounding difference (the
// standard trade a parallel reduction makes; matches numpy/BLAS/torch reductions).
func SumF64(n int, chunkSum func(lo, hi int) float64) float64 {
	parts := min(len(mailboxes)+1, n)
	if parts <= 1 {
		return chunkSum(0, n)
	}
	chunk := (n + parts - 1) / parts
	partials := make([]float64, parts) // chunk c (lo == c*chunk) writes partials[c]; unused slots stay 0
	Rows(n, func(lo, hi int) { partials[lo/chunk] = chunkSum(lo, hi) })
	var s float64
	for _, p := range partials {
		s += p
	}
	return s
}

// SumF64x2 is [SumF64] for a chunkSum that produces TWO partial sums per chunk — the fused
// map-reduce an optimizer needs when a single memory pass both updates per-index state (a disjoint
// write, bit-identical to serial) and accumulates two reductions over it. It splits [0,n) exactly as
// [Rows] does, calls chunkSum once per chunk (concurrently, disjoint ranges), and combines each
// partial in the SAME fixed chunk order, so both totals are deterministic REASSOCIATIONS of the
// serial left-to-right sums (~1-ULP-per-level tolerance, as SumF64).
func SumF64x2(n int, chunkSum func(lo, hi int) (float64, float64)) (float64, float64) {
	parts := min(len(mailboxes)+1, n)
	if parts <= 1 {
		return chunkSum(0, n)
	}
	chunk := (n + parts - 1) / parts
	p1 := make([]float64, parts) // chunk c (lo == c*chunk) writes p1[c]/p2[c]; unused slots stay 0
	p2 := make([]float64, parts)
	Rows(n, func(lo, hi int) { p1[lo/chunk], p2[lo/chunk] = chunkSum(lo, hi) })
	var s1, s2 float64
	for i := range p1 {
		s1 += p1[i]
		s2 += p2[i]
	}
	return s1, s2
}
