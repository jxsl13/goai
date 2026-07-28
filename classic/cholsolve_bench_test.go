package classic_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/classic"
)

// LinearRegression.Fit routes through cholSolve on the normal equations, so this
// exercises the Cholesky factor whose layout is under test. Feature count drives n
// (the matrix side), which is what the O(n³) factorization scales with.
func benchOLS(b *testing.B, n, feat int) {
	rng := rand.New(rand.NewSource(1))
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := range x {
		row := make([]float64, feat)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		x[i], y[i] = row, rng.NormFloat64()
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		m := &classic.LinearRegression{}
		if err := m.Fit(x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkOLSFit_512x64(b *testing.B) { benchOLS(b, 512, 64) }
func BenchmarkOLSFit_256x16(b *testing.B) { benchOLS(b, 256, 16) }
