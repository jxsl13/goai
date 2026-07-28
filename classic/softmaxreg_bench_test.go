package classic_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/classic"
)

// SoftmaxRegression.Fit runs a Newton loop with an Armijo line search; the
// per-step line-search scratch is what is under test, so allocs/op is the primary
// metric. More steps means more of that scratch, so the two step counts separate
// the per-step cost from the fixed setup.
func benchSoftmaxFit(b *testing.B, n, feat, k, steps int) {
	rng := rand.New(rand.NewSource(1))
	x := make([][]float64, n)
	y := make([]int, n)
	for i := range x {
		row := make([]float64, feat)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		x[i], y[i] = row, rng.Intn(k)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		m := &classic.SoftmaxRegression{}
		if err := m.Fit(x, y, k, steps, 0.1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSoftmaxFit_512x16x4_s50(b *testing.B)  { benchSoftmaxFit(b, 512, 16, 4, 50) }
func BenchmarkSoftmaxFit_512x16x4_s200(b *testing.B) { benchSoftmaxFit(b, 512, 16, 4, 200) }
