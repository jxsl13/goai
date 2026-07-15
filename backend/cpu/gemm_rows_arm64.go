//go:build goexperiment.simd

package cpu

// gemmF32Rows computes C rows [loRow,hiRow) of C[m,n] = A[m,k]·B[k,n],
// f32-native, store semantics — the WORKER-CALLABLE band entry the gemm-routed
// MHA/Conv kernels use inside a single parallelWork pass (no nested pool
// fork/join; the caller owns the parallelism). On arm64+simd it is the ADR-0026
// NEON tile band (unpacked B; the B panels of the attention/conv shapes are
// L2-resident).
func gemmF32Rows(A, B, C []float32, loRow, hiRow, k, n int) {
	if k == 0 { // the asm k-loop is do-while
		for i := loRow; i < hiRow; i++ {
			clear(C[i*n : (i+1)*n])
		}
		return
	}
	gemmF32BandNeon(A, B, nil, C, loRow, hiRow, k, n)
}

// gemmF32RowsCols is gemmF32Rows restricted to columns [jLo,jHi) — the causal
// attention bands use it to skip the fully-masked column span. jLo must be
// 16-aligned (the NEON cols kernel's contract; callers pass 0).
func gemmF32RowsCols(A, B, C []float32, loRow, hiRow, k, n, jLo, jHi int) {
	if k == 0 {
		for i := loRow; i < hiRow; i++ {
			clear(C[i*n+jLo : i*n+jHi])
		}
		return
	}
	gemmF32BandNeonCols(A, B, nil, C, loRow, hiRow, k, n, jLo, jHi)
}
