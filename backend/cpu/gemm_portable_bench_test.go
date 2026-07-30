//go:build !(amd64 && goexperiment.simd) && !(arm64 && goexperiment.simd)

package cpu

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkGemmF32Portable exercises the portable f32 GEMM — gemmF32 over the scalar band
// kernels in gemm_nosimd.go — which had no benchmark on this build configuration.
//
// The existing head-to-head in gemm_amx_bench_test.go is behind //go:build goexperiment.simd, so on
// a default build (this host: darwin/arm64, no experiment) nothing measured the kernel that
// actually runs, even though it is 15.95% of the vision benchmark suite's profile.
//
// The sweep spans the point where the accumulator stops fitting in cache: the band kernel holds 4
// rows of the f64 accumulator live across a whole k pass, which is 4*n*8 bytes — 8KB at n=256, 32KB
// at n=1024.
func BenchmarkGemmF32Portable(b *testing.B) {
	for _, sz := range []int{64, 128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("%d", sz), func(b *testing.B) {
			m, k, n := sz, sz, sz
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
			// Sweep the pack gate inside ONE binary, as the f64 benchmark does: the arms then
			// differ by that variable alone. This is also how the gate itself is calibrated, and
			// it has to be re-run whenever the packed band changes — a threshold measured against
			// an older, slower packed kernel is stale by construction.
			for _, arm := range []struct {
				name string
				gate int
			}{{"unpacked", 1 << 30}, {"packed", 0}} {
				b.Run(arm.name, func(b *testing.B) {
					saved := gemmPackMinWorkF32
					gemmPackMinWorkF32 = arm.gate
					defer func() { gemmPackMinWorkF32 = saved }()
					b.ResetTimer()
					for range b.N {
						gemmF32(A, B, C, m, k, n)
					}
					b.StopTimer()
					flops := 2 * float64(m) * float64(k) * float64(n)
					b.ReportMetric(flops/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
				})
			}
		})
	}
}

// BenchmarkGemmF64Portable is the f64 twin, driven the way gemm.go drives it for a large matmul
// (parallelWork over row bands). The f64 band kernel is also reached by the conv forward and
// backward im2col paths, so it is not only the F64 matmul that pays for it.
//
// C accumulates across iterations rather than being cleared, which is what this kernel's contract
// says it does; the values stay far from overflow at these sizes and the arithmetic cost does not
// depend on them.
func BenchmarkGemmF64Portable(b *testing.B) {
	for _, sz := range []int{64, 128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("%d", sz), func(b *testing.B) {
			m, k, n := sz, sz, sz
			rng := rand.New(rand.NewSource(7))
			A := make([]float64, m*k)
			Bm := make([]float64, k*n)
			C := make([]float64, m*n)
			for i := range A {
				A[i] = rng.Float64()*2 - 1
			}
			for i := range Bm {
				Bm[i] = rng.Float64()*2 - 1
			}
			// Drive gemmF64Rows, which is what gemm.go calls for a large matmul, and sweep the
			// pack gate inside ONE binary: the arms then differ only by that variable, with no
			// file swap and no recompilation between them.
			for _, arm := range []struct {
				name string
				gate int
			}{{"unpacked", 1 << 30}, {"packed", 0}} {
				b.Run(arm.name, func(b *testing.B) {
					saved := gemmPackMinWorkF64
					gemmPackMinWorkF64 = arm.gate
					defer func() { gemmPackMinWorkF64 = saved }()
					b.ResetTimer()
					for range b.N {
						gemmF64Rows(A, Bm, C, m, k, n)
					}
					b.StopTimer()
					flops := 2 * float64(m) * float64(k) * float64(n)
					b.ReportMetric(flops/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
				})
			}
		})
	}
}
