package classic

import (
	"math/rand/v2"
	"testing"
)

// svmBenchData builds a fixed synthetic binary dataset mirroring the §V22
// perf-parity probe (Gaussian blobs, one class vs. the rest) at a given size.
func svmBenchData(n, d int) ([][]float64, []float64) {
	const classes = 3
	rng := rand.New(rand.NewPCG(42, 42))
	centers := make([][]float64, classes)
	for c := range centers {
		centers[c] = make([]float64, d)
		for j := range centers[c] {
			centers[c][j] = rng.NormFloat64() * 3
		}
	}
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		c := rng.IntN(classes)
		if c != 0 {
			y[i] = 1
		}
		X[i] = make([]float64, d)
		for j := range X[i] {
			X[i][j] = centers[c][j] + rng.NormFloat64()
		}
	}
	return X, y
}

// BenchmarkSVCFit times the RBF SVC fit at a couple of sizes so the SMO
// optimization can be A/B'd (§V22). Run: go test ./classic/ -run x -bench BenchmarkSVCFit
func BenchmarkSVCFit(b *testing.B) {
	for _, n := range []int{1000, 4000} {
		X, y := svmBenchData(n, 20)
		b.Run(svmBenchName(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := NewSVC(WithSVMKernel(SVMKernelRBF)).Fit(X, y); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func svmBenchName(n int) string {
	switch n {
	case 1000:
		return "n1000_rbf"
	case 4000:
		return "n4000_rbf"
	default:
		return "n_rbf"
	}
}

func BenchmarkSVCPredict(b *testing.B) {
	X, y := svmBenchData(4000, 20)
	m := NewSVC(WithSVMKernel(SVMKernelRBF))
	if err := m.Fit(X, y); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.DecisionFunction(X); err != nil {
			b.Fatal(err)
		}
	}
}
