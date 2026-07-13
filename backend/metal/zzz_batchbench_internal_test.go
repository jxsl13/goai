//go:build darwin && cgo

package metal

import (
	"testing"
	"time"
)

// §T368 (§V22): validate ADR-0019's core premise — does batching N dispatches into ONE
// command buffer (one submit/wait) recover the per-op round-trip overhead? A decode step
// is ~95 ops; if batching is ~N× faster here, the batched-decode path (ADR-0019 Phase 2)
// is worth building; if not, park it.
func TestBatchDispatchBench(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const iters = 95
	timeit := func(one bool) float64 {
		batchDispatchBench(iters, one)
		const N = 30
		s := time.Now()
		for i := 0; i < N; i++ {
			batchDispatchBench(iters, one)
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	perOp := timeit(false)
	batched := timeit(true)
	t.Logf("%d dispatches: per-op %.2f ms (%.0f us/op) | batched(1 cmd-buffer) %.2f ms | speedup %.1fx",
		iters, perOp, perOp*1000/iters, batched, perOp/batched)
}
