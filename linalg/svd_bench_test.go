package linalg_test

import (
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSVDPCA times the one-sided Jacobi SVD on a tall [2000,50] matrix — the
// PCA / low-rank regime (m≫n) where the working-buffer column access dominates.
// Deterministic xorshift fill (no math/rand) so the A/B is stable across runs.
func BenchmarkSVDPCA(b *testing.B) {
	const m, n = 2000, 50
	a := tensor.New(tensor.F64, tensor.Shape{m, n})
	x := uint64(0x9e3779b97f4a7c15)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			a.SetF64(float64(int64(x))/(1<<62), i, j)
		}
	}
	b.ResetTimer()
	for range b.N {
		u, s, v, err := linalg.SVD(a)
		if err != nil {
			b.Fatal(err)
		}
		_, _, _ = u, s, v
	}
}

// benchSVDShape times SVD on an m×n matrix (deterministic xorshift fill). Square shapes
// exercise the V-accumulator rotation (n-length, the flat-buffer path); the tall control
// shows the win holds when the dot-products dominate.
func benchSVDShape(b *testing.B, m, n int) {
	a := tensor.New(tensor.F64, tensor.Shape{m, n})
	x := uint64(0x9e3779b97f4a7c15)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			a.SetF64(float64(int64(x))/(1<<62), i, j)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, _, err := linalg.SVD(a); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSVD_128x128(b *testing.B) { benchSVDShape(b, 128, 128) }
func BenchmarkSVD_192x192(b *testing.B) { benchSVDShape(b, 192, 192) }
func BenchmarkSVD_256x64(b *testing.B)  { benchSVDShape(b, 256, 64) }
