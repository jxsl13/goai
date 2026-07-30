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

// benchPredictProbaD parametrizes the feature dimension: the fused density kernel
// reuses each x[j] load across 4 components, so its advantage over per-component
// scalar logGaussian grows with d (the 2-feature cases above understate it).
func benchPredictProbaD(b *testing.B, n, k, d int) {
	rng := rand.New(rand.NewSource(1))
	x := make([][]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		x[i] = row
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

func BenchmarkGMMPredictProba_512x8_d16(b *testing.B) { benchPredictProbaD(b, 512, 8, 16) }
func BenchmarkGMMPredictProba_512x8_d32(b *testing.B) { benchPredictProbaD(b, 512, 8, 32) }
func BenchmarkGMMPredictProba_512x8_d64(b *testing.B) { benchPredictProbaD(b, 512, 8, 64) }

func benchPredictProbaFull(b *testing.B, n, k, d int) {
	rng := rand.New(rand.NewSource(1))
	x := make([][]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			row[j] = rng.NormFloat64() + float64(i%k)
		}
		x[i] = row
	}
	m := classic.NewGaussianMixture(classic.WithGMMComponents(k), classic.WithGMMCovariance(classic.GMMFull), classic.WithGMMSeed(1))
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

func BenchmarkGMMPredictProbaFull_2048x8_d20(b *testing.B) { benchPredictProbaFull(b, 2048, 8, 20) }
