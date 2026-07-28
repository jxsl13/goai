package classic_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/classic"
)

// PredictProba over many samples: the per-sample scratch allocation is the thing
// under test, so allocs/op and B/op are the primary metrics, not ns/op.
func benchPredictProba(b *testing.B, n, k int) {
	rng := rand.New(rand.NewSource(1))
	x := make([][]float64, n)
	for i := range x {
		x[i] = []float64{rng.NormFloat64(), rng.NormFloat64()}
	}
	m := classic.NewGaussianMixture(classic.WithGMMComponents(k), classic.WithGMMSeed(1))
	if err := m.Fit(x); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := m.PredictProba(x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGMMPredictProba_2048x8(b *testing.B) { benchPredictProba(b, 2048, 8) }
func BenchmarkGMMPredictProba_512x4(b *testing.B)  { benchPredictProba(b, 512, 4) }
func BenchmarkGMMPredictProba_512x8(b *testing.B)  { benchPredictProba(b, 512, 8) }
func BenchmarkGMMPredictProba_2048x4(b *testing.B) { benchPredictProba(b, 2048, 4) }
