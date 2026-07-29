package classic

import (
	"math/rand/v2"
	"testing"
)

func benchGBMSetup(nTrain, nTest, nFeat, nTrees int) (*GradientBoostingRegressor, [][]float64) {
	rng := rand.New(rand.NewPCG(1, 2))
	xtr := make([][]float64, nTrain)
	y := make([]float64, nTrain)
	for i := range xtr {
		row := make([]float64, nFeat)
		var s float64
		for j := range row {
			row[j] = rng.NormFloat64()
			s += row[j]
		}
		xtr[i] = row
		y[i] = s + rng.NormFloat64()*0.1
	}
	m := NewGradientBoostingRegressor(WithGBMNEstimators(nTrees), WithGBMLearningRate(0.1), WithGBMMaxDepth(3))
	if err := m.Fit(xtr, y); err != nil {
		panic(err)
	}
	xte := make([][]float64, nTest)
	for i := range xte {
		row := make([]float64, nFeat)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		xte[i] = row
	}
	return m, xte
}

func BenchmarkGBMPredictBatch(b *testing.B) {
	m, xte := benchGBMSetup(2000, 8192, 16, 100)
	b.ResetTimer()
	for range b.N {
		if _, err := m.Predict(xte); err != nil {
			b.Fatal(err)
		}
	}
}
