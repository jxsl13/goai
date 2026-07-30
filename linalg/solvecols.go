package linalg

import (
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
