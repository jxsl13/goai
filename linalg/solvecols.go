package linalg

import (
	"github.com/jxsl13/goai/internal/parallel"
	"runtime"
	"sync"
)

// solveColsThreshold is the total work (n² per column) below which splitting the
// right-hand-side columns costs more than it saves.
const solveColsThreshold = 1 << 15

// solveCols runs body over disjoint ranges of the cols right-hand sides, handing each worker
// its OWN scratch buffer of length scratch. Serial below a work threshold and for a single
// worker.
//
// Per-worker scratch is what makes these solves concurrent at all. Each triangular solve needs
// a forward-substitution buffer; hoisting one out of the column loop removed cols allocations
// per call but coupled every column to every other, so the loop could not fan out. A GOMAXPROCS
// sweep measured the whole linalg package at 1.00-1.10x from one core to twelve — the
// factorizations are inherently sequential, but the SOLVE phase over many right-hand sides is
// embarrassingly parallel and was being left on one core.
//
// BIT-IDENTICAL by construction: column c writes only out[i*cols+c], reads only its own
// scratch, and its substitution order is untouched. No accumulation crosses columns, so
// partitioning them cannot change a value.
func solveCols(cols, per, scratch int, body func(lo, hi int, buf []float64)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > cols {
		nw = cols
	}
	if nw <= 1 || cols*per < solveColsThreshold {
		body(0, cols, make([]float64, scratch))
		return
	}
	csz := (cols + nw - 1) / nw
	var wg sync.WaitGroup
	for c := 0; c < nw; c++ {
		lo := c * csz
		if lo >= cols {
			break
		}
		hi := min(lo+csz, cols)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi, make([]float64, scratch))
		}(lo, hi)
	}
	wg.Wait()
}

// factorParThreshold is the trailing-submatrix element count below which splitting an
// elimination step costs more than it saves. A factorization runs n steps whose work shrinks
// quadratically, so most of the later steps fall under this and stay serial.
//
// A var, not a const, so a parity test can raise it to force the serial arm and diff the two
// arms against each other. That is not a cosmetic difference: every golden in this package ran
// at a geometry FAR below this threshold (QR's is 24x16 = 384 elements against a bound of
// 65536), so the parallel arms had no bit-identity coverage at all — the goldens were pinning
// the serial path and reading as though they pinned both. Treat as read-only outside tests.
var factorParThreshold = 1 << 16

// parallelRowsIf runs body over disjoint ranges of [0,n) when want is true, and inline
// otherwise. Callers decide the predicate because the work per step varies.
//
// It routes through the SHARED BOUNDED POOL rather than spawning. A factorization calls this
// once per elimination step — 768 of them at n=768 — and spawning a worker set per step took
// Inverse from 823 allocations to 17463, a 21x regression on the resource axis that a 1.74x
// wall-clock win does not excuse. The pool's workers already exist.
func parallelRowsIf(want bool, n int, body func(lo, hi int)) {
	if !want || n < 2 || parallel.Workers() <= 1 {
		body(0, n)
		return
	}
	parallel.Rows(n, body)
}
