package classic_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/classic"
)

func forestFixture(tb testing.TB) (*classic.RandomForestClassifier, [][]float64) {
	const nTrain, nFeat = 400, 12
	X := make([][]float64, nTrain)
	y := make([]int, nTrain)
	for i := range X {
		X[i] = make([]float64, nFeat)
		s := 0.0
		for j := range X[i] {
			X[i][j] = math.Sin(float64(i*7 + j))
			s += X[i][j]
		}
		if s > 0 {
			y[i] = 1
		}
	}
	m := classic.NewRandomForestClassifier(classic.WithNumTrees(100), classic.WithSeed(1))
	if err := m.Fit(X, y); err != nil {
		tb.Fatal(err)
	}
	Xq := make([][]float64, 800)
	for i := range Xq {
		Xq[i] = X[i%nTrain]
	}
	return m, Xq
}

func BenchmarkForestPredict(b *testing.B) {
	m, Xq := forestFixture(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := m.Predict(Xq); err != nil {
			b.Fatal(err)
		}
	}
}
