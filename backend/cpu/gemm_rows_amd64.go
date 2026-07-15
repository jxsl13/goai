//go:build goexperiment.simd

package cpu

// gemmF32Rows computes C rows [loRow,hiRow) of C[m,n] = A[m,k]·B[k,n],
// f32-native, store semantics — the WORKER-CALLABLE band entry the gemm-routed
// MHA/Conv kernels use inside a single parallelWork pass (no nested pool
// fork/join; the caller owns the parallelism). On amd64+simd it is the
// ADR-0021 AVX band kernel; without FMA it falls back to the portable
// f32-native scalar rows (still tolerance-grade, never bit-exactness-gated —
// the f32NativeKernels const only routes here on tolerant builds).
func gemmF32Rows(A, B, C []float32, loRow, hiRow, k, n int) {
	if !gemmHasFMA {
		gemmF32RowsPortable(A, B, C, loRow, hiRow, k, n)
		return
	}
	gemmF32BandDirect(A, B, C, loRow, hiRow, k, n)
}

// gemmF32RowsCols is gemmF32Rows restricted to columns [jLo,jHi) — the causal
// attention bands use it to skip the fully-masked column span.
func gemmF32RowsCols(A, B, C []float32, loRow, hiRow, k, n, jLo, jHi int) {
	if !gemmHasFMA {
		gemmF32RowsColsPortable(A, B, C, loRow, hiRow, k, n, jLo, jHi)
		return
	}
	gemmF32BandDirectCols(A, B, C, loRow, hiRow, k, n, jLo, jHi)
}
