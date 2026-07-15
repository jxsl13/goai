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

// f32Scratch pools the transient f32 buffers of the gemm-routed f32-native
// MHA/Conv paths (f32NativeKernels builds). Same zero-on-get contract as
// f64Scratch: im2col leaves padding entries untouched and the attention
// score buffer relies on masked columns staying zero.
var f32Scratch = sync.Pool{New: func() any { b := make([]float32, 0); return &b }}

// getF32 returns a zeroed []float32 of length n from the pool.
func getF32(n int) *[]float32 {
	bp := f32Scratch.Get().(*[]float32)
	if cap(*bp) < n {
		*bp = make([]float32, n)
	} else {
		*bp = (*bp)[:n]
		clear(*bp)
	}
	return bp
}

// putF32 returns a buffer to the pool.
func putF32(bp *[]float32) { f32Scratch.Put(bp) }

// getF32Raw returns a []float32 of length n from the pool WITHOUT zeroing —
// for buffers every element of which is overwritten before use (packed
// operands, store-semantics gemm outputs). Same pool as getF32.
func getF32Raw(n int) *[]float32 {
	bp := f32Scratch.Get().(*[]float32)
	if cap(*bp) < n {
		*bp = make([]float32, n)
	} else {
		*bp = (*bp)[:n]
	}
	return bp
}
