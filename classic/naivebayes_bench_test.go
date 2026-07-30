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

// BenchmarkGaussianNBFit times the FIT path, which nothing measured: BenchmarkGaussianNBPredict
// calls Fit in setup before ResetTimer, so every cost inside it was invisible. Two sizes, because
// the epsilon prepass is O(n*d) with a column-strided access pattern and its share should grow with
// the row count rather than stay fixed.
func BenchmarkGaussianNBFit(b *testing.B) {
	for _, g := range []struct{ n, d int }{{4000, 20}, {16000, 20}} {
		X, y := spatialData(g.n, g.d)
		b.Run(nbFitName(g.n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				m := NewGaussianNB()
				if err := m.Fit(X, y); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func nbFitName(n int) string {
	if n == 0 {
		return "n0"
	}
	var d [8]byte
	i := len(d)
	for n > 0 {
		i--
		d[i] = byte('0' + n%10)
		n /= 10
	}
	return "n" + string(d[i:])
}
