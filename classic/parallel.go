package classic

import (
	"runtime"
	"sync"
)

// classicBandGrain is the element-work one worker must be given before another is added. The
// SVC kernel cache calls parallelBands thousands of times per fit with a column that is only
// just above the gate, which is the regime where a fixed GOMAXPROCS split spends more on
// wakeups than the split saves (§SIZE-THE-FANOUT-TO-THE-WORK-001).
const classicBandGrain = 1 << 13

// parallelBands runs body over disjoint bands of [0,n), sized to the work rather than only
// gated on it, and runs it inline in one call below a threshold. The CALLER owns disjointness:
// the helper guarantees only that the bands it hands out do not overlap.
func parallelBands(n, workPerItem int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > n {
		nw = n
	}
	if w := n * workPerItem / classicBandGrain; w < nw {
		nw = w
	}
	if nw <= 1 || n < 2 {
		body(0, n)
		return
	}
	var wg sync.WaitGroup
	//perfscan:ignore PS3011 one-time chunk-size calc in fan-out helper, already work-sized
	chunk := (n + nw - 1) / nw
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}
