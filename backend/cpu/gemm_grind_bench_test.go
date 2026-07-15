package cpu

import (
	"math/rand"
	"testing"
)

// Direct-kernel GEMM benchmarks (no tensor/dispatch overhead) over square and
// LLM-representative rectangular shapes. 2·M·K·N flops per op; reported as the
// GFLOP/s custom metric so deltas read directly in throughput.

func benchGemmF32Direct(b *testing.B, m, k, n int) {
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
		gemmF32(A, B, C, m, k, n)
	}
	b.StopTimer()
	flops := 2 * float64(m) * float64(k) * float64(n)
	b.ReportMetric(flops/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

func benchGemmF64Direct(b *testing.B, m, k, n int) {
	rng := rand.New(rand.NewSource(1))
	A := make([]float64, m*k)
	B := make([]float64, k*n)
	C := make([]float64, m*n)
	for i := range A {
		A[i] = rng.Float64() - 0.5
	}
	for i := range B {
		B[i] = rng.Float64() - 0.5
	}
	b.ResetTimer()
	for range b.N {
		clear(C) // += contract: zero C each iteration
		parallelWork(m, k*n, func(lo, hi int) {
			gemmF64Band(A, B, C, lo, hi, k, n)
		})
	}
	b.StopTimer()
	flops := 2 * float64(m) * float64(k) * float64(n)
	b.ReportMetric(flops/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

func BenchmarkGemmDirF32_256(b *testing.B)  { benchGemmF32Direct(b, 256, 256, 256) }
func BenchmarkGemmDirF32_512(b *testing.B)  { benchGemmF32Direct(b, 512, 512, 512) }
func BenchmarkGemmDirF32_1024(b *testing.B) { benchGemmF32Direct(b, 1024, 1024, 1024) }

// LLM-ish shapes: token-batch × hidden GEMMs.
func BenchmarkGemmDirF32_512x2048x2048(b *testing.B) { benchGemmF32Direct(b, 512, 2048, 2048) }
func BenchmarkGemmDirF32_512x2048x8192(b *testing.B) { benchGemmF32Direct(b, 512, 2048, 8192) }
func BenchmarkGemmDirF32_1x2048x2048(b *testing.B)   { benchGemmF32Direct(b, 1, 2048, 2048) } // decode GEMV shape

// Odd shapes exercising tails (n%16, n%8, n%4, row remainder).
func BenchmarkGemmDirF32_511x513x515(b *testing.B) { benchGemmF32Direct(b, 511, 513, 515) }

func BenchmarkGemmDirF64_512(b *testing.B)           { benchGemmF64Direct(b, 512, 512, 512) }
func BenchmarkGemmDirF64_1024(b *testing.B)          { benchGemmF64Direct(b, 1024, 1024, 1024) }
func BenchmarkGemmDirF64_512x2048x2048(b *testing.B) { benchGemmF64Direct(b, 512, 2048, 2048) }
