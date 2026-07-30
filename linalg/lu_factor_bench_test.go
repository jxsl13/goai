package linalg_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// Factor is the O(n³) partial-pivoting LU — the dominant cost of Solve/Inverse/Det.
// Measures the flat row-major working copy vs the prior [][]float64 (scattered rows).
func benchFactor(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(7))
	ad := make([]float64, n*n)
	for i := range ad {
		ad[i] = rng.NormFloat64()
	}
	for i := range n { // diagonally dominant → well-conditioned, no degenerate pivots
		ad[i*n+i] += float64(n)
	}
	a := tensor.FromFloat64(tensor.Shape{n, n}, ad)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := linalg.Factor(a); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFactor128(b *testing.B) { benchFactor(b, 128) }
func BenchmarkFactor256(b *testing.B) { benchFactor(b, 256) }
func BenchmarkFactor512(b *testing.B) { benchFactor(b, 512) }
