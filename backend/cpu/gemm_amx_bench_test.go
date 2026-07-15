//go:build darwin && arm64 && goexperiment.simd

package cpu

// ADR-0027 head-to-head: NEON (ADR-0026, pure Go) vs raw-AMX Plan9 asm
// (Path B, pure Go) vs Accelerate cblas_sgemm (Path A, cgo) at the
// torch-compared shapes. Run with:
//
//	GOEXPERIMENT=simd CGO_ENABLED=1 go test ./backend/cpu/ -run XXX \
//	    -bench BenchmarkGemmF32Path -count=6
//
// Measured (M2 Pro 8P+4E, §V22 paired A/B medians, GFLOP/s; torch-cpu bar
// 2584 — sustained sweeps throttle ~10%, so cross-check marginal winners
// with alternating -count=1 pairs):
//
//	shape            NEON  rawAMX  Accel   winner (dispatched)
//	256³              428     537   1116   Accel
//	512³              707    1524   2210   Accel
//	1024³             758   ~2100  ~2590   Accel (≈1.0× torch-cpu)
//	512×2048×2048     656    1878   1786   raw AMX (+5%)
//	2048³               —    2325   2100   raw AMX (+11%)
//	512×2048×4096       —    1695   1294   raw AMX (+31%)
//	512×2048×8192     550    1570   1610   raw AMX (tie, −3% ≈ noise)
//
// The dispatch in gemmF32 (gemm_neon_arm64.go) encodes this: raw AMX for
// k·n·4 ≥ 16 MiB && m ≥ 256, Accelerate otherwise, NEON when neither fits
// the build/host. Full tables in ADR-0027 and docs/benchmarking.md.

import (
	"math/rand"
	"testing"
)

func benchGemmPath(b *testing.B, path string, m, k, n int) {
	savedAccel, savedAMX := gemmF32Accel, gemmF32AMX
	defer func() { gemmF32Accel, gemmF32AMX = savedAccel, savedAMX }()
	switch path {
	case "neon":
		gemmF32Accel, gemmF32AMX = nil, nil
	case "amx":
		if savedAMX == nil {
			b.Skip("no AMX on this host")
		}
		gemmF32Accel, gemmF32AMX = nil, savedAMX
	case "accel":
		if savedAccel == nil {
			b.Skip("Accelerate needs cgo")
		}
		gemmF32Accel, gemmF32AMX = savedAccel, nil
	}
	rng := rand.New(rand.NewSource(42))
	A := make([]float32, m*k)
	B := make([]float32, k*n)
	C := make([]float32, m*n)
	for i := range A {
		A[i] = rng.Float32()*2 - 1
	}
	for i := range B {
		B[i] = rng.Float32()*2 - 1
	}
	b.ResetTimer()
	for range b.N {
		gemmF32(A, B, C, m, k, n)
	}
	b.StopTimer()
	flops := 2 * float64(m) * float64(k) * float64(n)
	b.ReportMetric(flops/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

func BenchmarkGemmF32Path_NEON_256(b *testing.B)   { benchGemmPath(b, "neon", 256, 256, 256) }
func BenchmarkGemmF32Path_AMX_256(b *testing.B)    { benchGemmPath(b, "amx", 256, 256, 256) }
func BenchmarkGemmF32Path_Accel_256(b *testing.B)  { benchGemmPath(b, "accel", 256, 256, 256) }
func BenchmarkGemmF32Path_NEON_512(b *testing.B)   { benchGemmPath(b, "neon", 512, 512, 512) }
func BenchmarkGemmF32Path_AMX_512(b *testing.B)    { benchGemmPath(b, "amx", 512, 512, 512) }
func BenchmarkGemmF32Path_Accel_512(b *testing.B)  { benchGemmPath(b, "accel", 512, 512, 512) }
func BenchmarkGemmF32Path_NEON_1024(b *testing.B)  { benchGemmPath(b, "neon", 1024, 1024, 1024) }
func BenchmarkGemmF32Path_AMX_1024(b *testing.B)   { benchGemmPath(b, "amx", 1024, 1024, 1024) }
func BenchmarkGemmF32Path_Accel_1024(b *testing.B) { benchGemmPath(b, "accel", 1024, 1024, 1024) }

func BenchmarkGemmF32Path_NEON_512x2048x2048(b *testing.B) {
	benchGemmPath(b, "neon", 512, 2048, 2048)
}
func BenchmarkGemmF32Path_AMX_512x2048x2048(b *testing.B) { benchGemmPath(b, "amx", 512, 2048, 2048) }
func BenchmarkGemmF32Path_Accel_512x2048x2048(b *testing.B) {
	benchGemmPath(b, "accel", 512, 2048, 2048)
}
func BenchmarkGemmF32Path_NEON_512x2048x8192(b *testing.B) {
	benchGemmPath(b, "neon", 512, 2048, 8192)
}
func BenchmarkGemmF32Path_AMX_512x2048x8192(b *testing.B) { benchGemmPath(b, "amx", 512, 2048, 8192) }
func BenchmarkGemmF32Path_Accel_512x2048x8192(b *testing.B) {
	benchGemmPath(b, "accel", 512, 2048, 8192)
}

func BenchmarkGemmF32Path_AMX_512x2048x4096(b *testing.B) { benchGemmPath(b, "amx", 512, 2048, 4096) }
func BenchmarkGemmF32Path_Accel_512x2048x4096(b *testing.B) {
	benchGemmPath(b, "accel", 512, 2048, 4096)
}
func BenchmarkGemmF32Path_AMX_2048(b *testing.B)   { benchGemmPath(b, "amx", 2048, 2048, 2048) }
func BenchmarkGemmF32Path_Accel_2048(b *testing.B) { benchGemmPath(b, "accel", 2048, 2048, 2048) }
