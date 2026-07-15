//go:build darwin && cgo && arm64 && goexperiment.simd

package cpu

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cpu/internal/accel"
)

// §V22 A/B for the decode-GEMV dispatch (ADR-0027 follow-up): Accelerate
// cblas_sgemv per output row vs cblas_sgemm called at the same small-m shape.
// GEMV is bandwidth-bound — the whole k×n weight panel is read once per row —
// so GFLOP/s here reads directly as memory-bandwidth utilization.
func benchGemvPath(b *testing.B, path string, m, k, n int) {
	if accel.SGEMV == nil || accel.SGEMM == nil {
		b.Skip("accel unavailable")
	}
	rng := rand.New(rand.NewSource(1))
	A := make([]float32, m*k)
	B := make([]float32, k*n)
	C := make([]float32, m*n)
	for i := range A {
		A[i] = rng.Float32() - 0.5
	}
	for i := range B {
		B[i] = rng.Float32() - 0.5
	}
	b.ResetTimer()
	for range b.N {
		switch path {
		case "sgemv":
			for i := 0; i < m; i++ {
				accel.SGEMV(B, A[i*k:(i+1)*k], C[i*n:(i+1)*n], k, n)
			}
		case "sgemm":
			accel.SGEMM(A, B, C, m, k, n)
		case "scalar":
			gemmF32SmallMScalar(A, B, C, m, k, n)
		}
	}
	b.StopTimer()
	flops := 2 * float64(m) * float64(k) * float64(n)
	b.ReportMetric(flops/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

// The decode hot shape: m=1, hidden=2048.
func BenchmarkGemvF32_sgemv_1x2048x2048(b *testing.B)  { benchGemvPath(b, "sgemv", 1, 2048, 2048) }
func BenchmarkGemvF32_sgemm_1x2048x2048(b *testing.B)  { benchGemvPath(b, "sgemm", 1, 2048, 2048) }
func BenchmarkGemvF32_scalar_1x2048x2048(b *testing.B) { benchGemvPath(b, "scalar", 1, 2048, 2048) }

// FFN up / down and the vocab head.
func BenchmarkGemvF32_sgemv_1x2048x8192(b *testing.B)  { benchGemvPath(b, "sgemv", 1, 2048, 8192) }
func BenchmarkGemvF32_sgemm_1x2048x8192(b *testing.B)  { benchGemvPath(b, "sgemm", 1, 2048, 8192) }
func BenchmarkGemvF32_sgemv_1x8192x2048(b *testing.B)  { benchGemvPath(b, "sgemv", 1, 8192, 2048) }
func BenchmarkGemvF32_sgemm_1x8192x2048(b *testing.B)  { benchGemvPath(b, "sgemm", 1, 8192, 2048) }
func BenchmarkGemvF32_sgemv_1x2048x32000(b *testing.B) { benchGemvPath(b, "sgemv", 1, 2048, 32000) }
func BenchmarkGemvF32_sgemm_1x2048x32000(b *testing.B) { benchGemvPath(b, "sgemm", 1, 2048, 32000) }

// Dispatch-level: the full gemmF32 entry at the shapes the m<3 branch serves.
func BenchmarkGemvF32_dispatch_1x2048x2048(b *testing.B) { benchGemmF32Direct(b, 1, 2048, 2048) }
func BenchmarkGemvF32_dispatch_2x2048x2048(b *testing.B) { benchGemmF32Direct(b, 2, 2048, 2048) }

// The m=2..3 boundary: per-row sgemv re-reads the weight panel per row, sgemm
// reads it once — this decides where the m-branch cutoff sits.
func BenchmarkGemvF32_sgemv_2x2048x2048(b *testing.B) { benchGemvPath(b, "sgemv", 2, 2048, 2048) }
func BenchmarkGemvF32_sgemm_2x2048x2048(b *testing.B) { benchGemvPath(b, "sgemm", 2, 2048, 2048) }
func BenchmarkGemvF32_sgemv_3x2048x2048(b *testing.B) { benchGemvPath(b, "sgemv", 3, 2048, 2048) }
func BenchmarkGemvF32_sgemm_3x2048x2048(b *testing.B) { benchGemvPath(b, "sgemm", 3, 2048, 2048) }

// gemmF32SmallMScalar reproduces the pre-existing m<4 fallback (the column-
// parallel scalar path in gemmF32) for the A/B baseline, bypassing the
// dispatch so the comparison stays honest after the sgemv branch lands.
func gemmF32SmallMScalar(A, B, C []float32, m, k, n int) {
	workers := 12
	jblk := (n + workers - 1) / workers
	jblk = (jblk + 15) &^ 15
	jblk = min(max(jblk, 64), 512)
	blocks := (n + jblk - 1) / jblk
	parallelWork(blocks, m*k*jblk, func(loB, hiB int) {
		gemmF32RowsScalar(A, B, C, 0, m, k, n, loB*jblk, min(hiB*jblk, n))
	})
}
