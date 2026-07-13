package cpu

import "sync"

// f64Scratch pools the large transient buffers of the im2col conv kernels
// (§T463): profiling a real CNN training step showed ~70% of the wall time in
// runtime overhead — madvise churn from multi-MB per-call allocations and the
// resulting GC pressure — against ~10% in the actual GEMM. Buffers are ZEROED on
// get: the kernels rely on zero-initialized scratch (im2col leaves padding
// entries untouched, gemmF64Band accumulates with +=), and a pooled buffer
// carries stale data.
var f64Scratch = sync.Pool{New: func() any { b := make([]float64, 0); return &b }}

// getF64 returns a zeroed []float64 of length n from the pool.
func getF64(n int) *[]float64 {
	bp := f64Scratch.Get().(*[]float64)
	if cap(*bp) < n {
		*bp = make([]float64, n)
	} else {
		*bp = (*bp)[:n]
		clear(*bp)
	}
	return bp
}

// putF64 returns a buffer to the pool.
func putF64(bp *[]float64) { f64Scratch.Put(bp) }
