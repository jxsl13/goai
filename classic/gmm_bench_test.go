package classic

import "testing"

// BenchmarkGMMFit times the EM loop, whose E-step calls logGaussian for every
// (sample, component) each iteration — the diagonal Mahalanobis hot path.
func BenchmarkGMMFit(b *testing.B) {
	X, _ := spatialData(2000, 20)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewGaussianMixture(WithGMMComponents(8), WithGMMMaxIter(20), WithGMMSeed(1))
		if err := m.Fit(X); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGMMScore times ScoreSamples — pure per-(sample,component) logGaussian,
// where the Mahalanobis reciprocal is the dominant per-element cost.
func BenchmarkGMMScore(b *testing.B) {
	X, _ := spatialData(4000, 20)
	m := NewGaussianMixture(WithGMMComponents(8), WithGMMMaxIter(20), WithGMMSeed(1))
	if err := m.Fit(X); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.ScoreSamples(X); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGMMScoreFull times ScoreSamples with a full covariance — the Cholesky
// triangular solve per (sample, component), previously allocating y every call.
func BenchmarkGMMScoreFull(b *testing.B) {
	X, _ := spatialData(4000, 20)
	m := NewGaussianMixture(WithGMMComponents(8), WithGMMMaxIter(20), WithGMMSeed(1),
		WithGMMCovariance(GMMFull))
	if err := m.Fit(X); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.ScoreSamples(X); err != nil {
			b.Fatal(err)
		}
	}
}
