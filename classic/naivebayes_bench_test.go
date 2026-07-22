package classic

import "testing"

// BenchmarkGaussianNBPredict times the joint-log-likelihood hot loop (per sample:
// nClasses × nFeat Gaussian terms) over a 4000×20 dataset.
func BenchmarkGaussianNBPredict(b *testing.B) {
	X, y := spatialData(4000, 20)
	m := NewGaussianNB()
	if err := m.Fit(X, y); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Predict(X); err != nil {
			b.Fatal(err)
		}
	}
}
