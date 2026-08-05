package nn

import "github.com/jxsl13/goai/internal/parallel"

// parStepMinElems is the per-parameter element count above which an optimizer's element-wise
// update is fanned across cores. Below it the parameter fits cache and a single core already
// saturates its memory bandwidth, so the goroutine/WaitGroup fan-out would only add overhead;
// the threshold sits above L2 (2 MB of f64 state / 1 MB f32) so ONLY genuinely DRAM-resident
// parameters — the embedding, lm_head and FFN matrices that dominate an LLM's numel — take the
// parallel path, and small models stay serial (zero regression).
//
// The motivation: forward and backward already fan every GEMM across all cores (parallelWork),
// so a single-threaded optimizer step is amplified to a large fraction of the PARALLEL step wall
// time — parallelizing it is the bit-exact analog of PyTorch's fused/foreach optimizer.
var parStepMinElems = 1 << 18

// parStep runs body over the index range [0,n): fanned into disjoint contiguous chunks across the
// worker pool when n is large enough to pay, else inline on the caller. body MUST write only
// per-element (index-i) state — moments and the parameter at index i — so the partition is
// BIT-IDENTICAL to the serial loop: every element computes exactly what it would alone, and no
// value is accumulated across chunks (the chunk ORDER is unspecified).
func parStep(n int, body func(lo, hi int)) {
	if n >= parStepMinElems {
		parallel.Rows(n, body)
		return
	}
	body(0, n)
}
