//go:build !(amd64 && goexperiment.simd) && !(arm64 && goexperiment.simd)

package cpu

// gemmF32 computes C[m,n] = A·B, f32 — the portable default. Accumulates in an
// f64 scratch (§V10) then narrows once, so the result is BIT-IDENTICAL to the
// f64 reference (§V3, §V11 tol 0). The perf builds override this entry point
// with f32-NATIVE kernels: gemm_simd.go on amd64+GOEXPERIMENT=simd (ADR-0021)
// and gemm_neon_arm64.go on arm64+GOEXPERIMENT=simd (ADR-0026) — both within
// the ADR-0021 tolerance of ref, not bit-exact.
func gemmF32(A, B, C []float32, m, k, n int) {
	accP := getF64(m * n) // pooled zeroed f64 accumulation scratch (§V10, §T463)
	acc := *accP
	parallelWork(m, k*n, func(loRow, hiRow int) {
		gemmF32Band(A, B, acc, loRow, hiRow, k, n)
	})
	for i := range C {
		C[i] = float32(acc[i])
	}
	putF64(accP)
}
