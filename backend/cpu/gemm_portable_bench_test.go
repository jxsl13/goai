//go:build !(amd64 && goexperiment.simd) && !(arm64 && goexperiment.simd)

package cpu

import (
	"fmt"
	"math/rand"
	"runtime"
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

// BenchmarkGemmF32PortableRows sweeps the ROW count at a fixed, large k*n, which is the axis
// gemmPackMinRows gates and the one the square sweep above cannot see. Decode- and attention-shaped
// matmuls are exactly this: few rows against a wide, deep B.
func BenchmarkGemmF32PortableRows(b *testing.B) {
	const k, n = 512, 512
	for _, m := range []int{32, 64, 96, 192} {
		b.Run(fmt.Sprintf("m=%d", m), func(b *testing.B) {
			rng := rand.New(rand.NewSource(11))
			A := make([]float32, m*k)
			B := make([]float32, k*n)
			C := make([]float32, m*n)
			for i := range A {
				A[i] = rng.Float32()*2 - 1
			}
			for i := range B {
				B[i] = rng.Float32()*2 - 1
			}
			for _, arm := range []struct {
				name string
				rows int
			}{{"unpacked", 1 << 30}, {"packed", 0}} {
				b.Run(arm.name, func(b *testing.B) {
					saved := gemmPackTileBlocksF32
					gemmPackTileBlocksF32 = arm.rows
					defer func() { gemmPackTileBlocksF32 = saved }()
					b.ResetTimer()
					for range b.N {
						gemmF32(A, B, C, m, k, n)
					}
				})
			}
		})
	}
}

// BenchmarkGemmF64PortableRows is the f64 twin of the row sweep. The row gate is shared between the
// dtypes, so per PERF-CACHE-GATE-IS-PER-DTYPE-001 it has to be checked on both before being moved.
func BenchmarkGemmF64PortableRows(b *testing.B) {
	const k, n = 512, 512
	for _, m := range []int{256, 384, 512} {
		b.Run(fmt.Sprintf("m=%d", m), func(b *testing.B) {
			rng := rand.New(rand.NewSource(13))
			A := make([]float64, m*k)
			Bm := make([]float64, k*n)
			C := make([]float64, m*n)
			for i := range A {
				A[i] = rng.Float64()*2 - 1
			}
			for i := range Bm {
				Bm[i] = rng.Float64()*2 - 1
			}
			for _, arm := range []struct {
				name string
				rows int
			}{{"unpacked", 1 << 30}, {"packed", 0}} {
				b.Run(arm.name, func(b *testing.B) {
					saved := gemmPackTileBlocksF64
					gemmPackTileBlocksF64 = arm.rows
					defer func() { gemmPackTileBlocksF64 = saved }()
					b.ResetTimer()
					for range b.N {
						gemmF64Rows(A, Bm, C, m, k, n)
					}
				})
			}
		})
	}
}

// BenchmarkGemmF32PackGrid sweeps BOTH axes the pack gate reads — blocks per band and pack size —
// which the two one-dimensional sweeps could not separate. m is chosen so each band gets exactly
// the target number of 4-row tile blocks on this host, and k=n sets the pack size independently.
func BenchmarkGemmF32PackGrid(b *testing.B) {
	w := runtime.GOMAXPROCS(0)
	for _, sz := range []int{64, 128, 256, 512} {
		for _, blocks := range []int{1, 2, 4} {
			m := blocks * 4 * w
			k, n := sz, sz
			b.Run(fmt.Sprintf("kn=%d/blocks=%d", k*n, blocks), func(b *testing.B) {
				if got := gemmPackBandCount(m, k, n); got != blocks {
					b.Skipf("geometry gives %d blocks per band, wanted %d", got, blocks)
				}
				rng := rand.New(rand.NewSource(int64(sz*7 + blocks)))
				A := make([]float32, m*k)
				B := make([]float32, k*n)
				C := make([]float32, m*n)
				for i := range A {
					A[i] = rng.Float32()*2 - 1
				}
				for i := range B {
					B[i] = rng.Float32()*2 - 1
				}
				for _, arm := range []struct {
					name  string
					gate  int
					block int
				}{{"unpacked", 1 << 30, 1 << 30}, {"packed", 0, 0}} {
					b.Run(arm.name, func(b *testing.B) {
						sw, sb := gemmPackMinWorkF32, gemmPackTileBlocksF32
						gemmPackMinWorkF32, gemmPackTileBlocksF32 = arm.gate, arm.block
						defer func() { gemmPackMinWorkF32, gemmPackTileBlocksF32 = sw, sb }()
						b.ResetTimer()
						for range b.N {
							gemmF32(A, B, C, m, k, n)
						}
					})
				}
			})
		}
	}
}

// BenchmarkGemmF32PackSameBlocks holds the block count and the pack size FIXED and varies only the
// rows per band within one block's width. If the pack decision were a function of blocks and pack
// size alone, these would all move together.
func BenchmarkGemmF32PackSameBlocks(b *testing.B) {
	w := runtime.GOMAXPROCS(0)
	const k, n = 64, 64
	for _, rowsPerBand := range []int{4, 5, 6, 7} {
		m := rowsPerBand * w
		b.Run(fmt.Sprintf("rowsPerBand=%d", rowsPerBand), func(b *testing.B) {
			if got := gemmPackBandCount(m, k, n); got != 1 {
				b.Skipf("%d blocks per band, wanted 1", got)
			}
			rng := rand.New(rand.NewSource(int64(rowsPerBand)))
			A := make([]float32, m*k)
			B := make([]float32, k*n)
			C := make([]float32, m*n)
			for i := range A {
				A[i] = rng.Float32()*2 - 1
			}
			for i := range B {
				B[i] = rng.Float32()*2 - 1
			}
			for _, arm := range []struct {
				name        string
				gate, block int
			}{{"unpacked", 1 << 30, 1 << 30}, {"packed", 0, 0}} {
				b.Run(arm.name, func(b *testing.B) {
					sw, sb := gemmPackMinWorkF32, gemmPackTileBlocksF32
					gemmPackMinWorkF32, gemmPackTileBlocksF32 = arm.gate, arm.block
					defer func() { gemmPackMinWorkF32, gemmPackTileBlocksF32 = sw, sb }()
					b.ResetTimer()
					for range b.N {
						gemmF32(A, B, C, m, k, n)
					}
				})
			}
		})
	}
}
