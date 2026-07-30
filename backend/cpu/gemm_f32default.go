//go:build !(amd64 && goexperiment.simd) && !(arm64 && goexperiment.simd)

package cpu

// gemmF32 computes C[m,n] = A·B, f32 — the portable default. Accumulates in an
// f64 scratch (§V10) then narrows once, so the result is BIT-IDENTICAL to the
// f64 reference (§V3, §V11 tol 0). The perf builds override this entry point
// with f32-NATIVE kernels: gemm_simd.go on amd64+GOEXPERIMENT=simd (ADR-0021)
// and gemm_neon_arm64.go on arm64+GOEXPERIMENT=simd (ADR-0026) — both within
// the ADR-0021 tolerance of ref, not bit-exact.
// gemmPackMinRows is the row count above which gemmF32 packs B: the pack costs one k*n copy and is
// reused by m/4 row blocks, so its overhead is ~2/m, and 32 rows puts that near 6%.
const gemmPackMinRows = 32

// gemmPackMinWork is the B element count above which packing pays. BOTH gates are needed and this
// is the one that decides it in practice, because packing is a CACHE fix, not a row-reuse fix: it
// only wins once B is too big to stay resident while the tiles walk it. Measured at three square
// sizes, packed vs not:
//
//	n=256  (B 256KB)  489.2us -> 502.8us   +2.78%  (p=0.002)  — pack is pure overhead
//	n=512  (B 1MB)    2.867ms -> 2.881ms       ~    (p=0.093)  — break-even
//	n=1024 (B 4MB)    21.80ms -> 18.93ms  -13.16%  (p=0.002)
//
// So the threshold sits above 1MB of B and below 4MB: 1<<19 elements is 2MB. A var rather than a
// const so a parity test can force each arm and diff them.
var gemmPackMinWork = 1 << 19

func gemmF32(A, B, C []float32, m, k, n int) {
	accP := getF64(m * n) // pooled zeroed f64 accumulation scratch (§V10, §T463)
	acc := *accP
	// Pack B into 4-column panels once, shared read-only by every row band. The packed band
	// reads a tile's k values contiguously instead of striding by n, so a cache line is fully
	// used rather than a quarter used, and the panel stays resident across the p sweep.
	//
	// GATED ON m, because the copy is k*n and the work it serves is m*k*n: the pack only pays
	// once enough row blocks reuse it. Below the gate the unpacked band runs, which is what
	// shipped before this and is unchanged.
	if m >= gemmPackMinRows && n >= 4 && k*n >= gemmPackMinWork {
		packP := getF32Raw((n >> 2) * k * 4)
		pack := *packP
		packBTiles4(B, pack, k, n)
		parallelWork(m, k*n, func(loRow, hiRow int) {
			gemmF32BandPacked(A, B, pack, acc, loRow, hiRow, k, n)
		})
		putF32(packP)
	} else {
		parallelWork(m, k*n, func(loRow, hiRow int) {
			gemmF32Band(A, B, acc, loRow, hiRow, k, n)
		})
	}
	for i := range C {
		C[i] = float32(acc[i])
	}
	putF64(accP)
}
