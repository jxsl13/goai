package linalg

import (
	"math"
	"testing"
)

// SymEig had no benchmark, so its known column-walk cost (acknowledged in the comment on
// the flat-buffer rewrite) could not be acted on. It is O(n^3) per sweep and backs the
// SOAP, Shampoo and GaLore preconditioners plus classic PCA and GMM.
func symEigInput(n int) [][]float64 {
	a := make([][]float64, n)
	for i := range a {
		a[i] = make([]float64, n)
	}
	for i := range n {
		for j := i; j < n; j++ {
			x := math.Sin(float64(i*n+j)*0.013) + 0.5*math.Cos(float64(i+j)*0.7)
			a[i][j], a[j][i] = x, x // symmetric
		}
		a[i][i] += float64(n) // diagonally dominant → well-conditioned
	}
	return a
}

func benchSymEig(b *testing.B, n int) {
	a := symEigInput(n)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		SymEig(a)
	}
}

func BenchmarkSymEig_64(b *testing.B)  { benchSymEig(b, 64) }
func BenchmarkSymEig_128(b *testing.B) { benchSymEig(b, 128) }
