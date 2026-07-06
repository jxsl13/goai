package simd

import "testing"

// Ceiling measurement for the portable loops (§T11b): elementwise add moves
// 3×8 bytes per element (two loads, one store) — if the plain Go loop already
// runs at a large fraction of cache bandwidth, hand-written NEON cannot win
// meaningfully and the arm64 elementwise ceiling is proven by measurement.

func benchAdd(b *testing.B, n int) {
	dst := make([]float64, n)
	x := make([]float64, n)
	y := make([]float64, n)
	for i := range x {
		x[i] = float64(i)
		y[i] = float64(n - i)
	}
	b.SetBytes(int64(n * 24)) // 2 loads + 1 store, 8B each
	b.ResetTimer()
	for range b.N {
		AddF64(dst, x, y)
	}
}

func BenchmarkAddF64_4K(b *testing.B)   { benchAdd(b, 4096) }    // L1-resident
func BenchmarkAddF64_256K(b *testing.B) { benchAdd(b, 262144) }  // L2/SLC
