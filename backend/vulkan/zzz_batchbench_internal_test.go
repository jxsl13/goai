//go:build vulkan && cgo

package vulkan

import (
	"testing"
	"time"
)

// §T380 (§V22, [[gpu-ops-all-backends]]): mirror the metal §T368 de-risk on Vulkan/MoltenVK BEFORE
// porting the whole Recorder. MoltenVK is bandwidth-bound and some Metal wins did NOT replicate
// here (§B39/§B41 register blocking), so the command-buffer batching premise must be MEASURED on
// this backend, not assumed. If batching N dispatches into one command buffer (one submit +
// waitIdle, barriers between) beats N per-op submit/waitIdle, the Vulkan Recorder port is worth it.
func TestVulkanBatchDispatchBench(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const iters, n = 95, 4096 // ~a decode step's worth of dispatches
	timeit := func(one bool) float64 {
		if err := batchDispatchBench(iters, one, n); err != nil { // warmup
			t.Fatal(err)
		}
		const N = 20
		s := time.Now()
		for range N {
			if err := batchDispatchBench(iters, one, n); err != nil {
				t.Fatal(err)
			}
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	perOp := timeit(false)
	batched := timeit(true)
	t.Logf("vulkan %d dispatches: per-op %.2f ms (%.0f us/op) | batched(1 cmd-buffer) %.2f ms | speedup %.1fx",
		iters, perOp, perOp*1000/iters, batched, perOp/batched)
}
