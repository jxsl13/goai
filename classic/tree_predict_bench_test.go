package classic_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/classic"
)

// benchmarks the standalone DecisionTree Predict paths (the forest variants are already
// parallel; the single-tree Predict was serial). A deep tree over a wide batch is the
// realistic serving shape: each row is an independent root-to-leaf walk.
func makeTreeData(n, d int, seed int64) ([][]float64, []int, []float64) {
	rng := rand.New(rand.NewSource(seed))
	x := make([][]float64, n)
	yc := make([]int, n)
	yr := make([]float64, n)
	for i := range x {
		x[i] = make([]float64, d)
		var s float64
		for j := range x[i] {
			x[i][j] = rng.NormFloat64()
			s += x[i][j] * float64(j%3-1)
		}
		yr[i] = s + 0.1*rng.NormFloat64()
		yc[i] = int(math.Mod(math.Abs(s*3), 5)) // 5 classes
	}
	return x, yc, yr
}

func BenchmarkDecisionTreeClassifierPredict(b *testing.B) {
	x, yc, _ := makeTreeData(20000, 32, 1)
	m := classic.NewDecisionTreeClassifier(classic.WithMaxDepth(18))
	if err := m.Fit(x, yc); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Predict(x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecisionTreeRegressorPredict(b *testing.B) {
	x, _, yr := makeTreeData(20000, 32, 2)
	m := classic.NewDecisionTreeRegressor(classic.WithMaxDepth(18))
	if err := m.Fit(x, yr); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Predict(x); err != nil {
			b.Fatal(err)
		}
	}
}
