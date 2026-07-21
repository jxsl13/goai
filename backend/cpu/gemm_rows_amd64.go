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
	gemmF32RowsCols(A, B, C, loRow, hiRow, k, n, 0, n)
}

// gemmF32RowsCols is gemmF32Rows restricted to columns [jLo,jHi) — the causal
// attention bands use it to skip the fully-masked column span. jLo is 16-aligned.
// The full 6×16 sub-tiles run through the hand-asm microkernel (SERIAL — the caller
// owns the parallelism); the m%6 / n%16 edges take the archsimd band kernel.
func gemmF32RowsCols(A, B, C []float32, loRow, hiRow, k, n, jLo, jHi int) {
	if !gemmHasFMA {
		gemmF32RowsColsPortable(A, B, C, loRow, hiRow, k, n, jLo, jHi)
		return
	}
	if !gemmAsmF32Ok(hiRow-loRow, jHi-jLo) {
		gemmF32BandDirectCols(A, B, C, loRow, hiRow, k, n, jLo, jHi)
		return
	}
	m6end := loRow + (hiRow - loRow) - (hiRow-loRow)%6
	n16end := jLo + (jHi-jLo)&^15
	for i := loRow; i < m6end; i += 6 {
		aBase := &A[i*k]
		for j := jLo; j < n16end; j += 16 {
			gemmF32Tile6x16AVX2(aBase, &B[j], &C[i*n+j], k, k, n, n)
		}
	}
	if n16end < jHi { // right column strip, all rows
		gemmF32BandDirectCols(A, B, C, loRow, hiRow, k, n, n16end, jHi)
	}
	if m6end < hiRow && n16end > jLo { // bottom row strip, left cols
		gemmF32BandDirectCols(A, B, C, m6end, hiRow, k, n, jLo, n16end)
	}
}
