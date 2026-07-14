package simd

import "testing"

// f32 elementwise ceiling (§T11b). Companion to the AddF64 benches: run the
// SAME benchmark under the default build (scalar) and under GOEXPERIMENT=simd
// (archsimd AVX Float32x8) for the A/B — f32 is where the 8-wide vector should
// pull ahead of the 4-wide f64 path if the loop is not purely bandwidth-bound.

func benchAddF32(b *testing.B, n int) {
	dst := make([]float32, n)
	x := make([]float32, n)
	y := make([]float32, n)
	for i := range x {
		x[i] = float32(i)
		y[i] = float32(n - i)
	}
	b.SetBytes(int64(n * 12)) // 2 loads + 1 store, 4B each
	b.ResetTimer()
	for range b.N {
		AddF32(dst, x, y)
	}
}

func BenchmarkAddF32_4K(b *testing.B)   { benchAddF32(b, 4096) }   // L1-resident
func BenchmarkAddF32_256K(b *testing.B) { benchAddF32(b, 262144) } // L2/SLC
