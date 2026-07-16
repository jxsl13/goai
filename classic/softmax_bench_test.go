package classic

import (
	"math"
	"math/rand"
	"testing"
)

// softmaxSynthetic builds a deterministic n×d, k-class dataset of k Gaussian
// blobs with a linearly-informative (but not separable) structure — the same
// shape as the §V22 A/B workload (n=4000, d=20, 3-class).
func softmaxSynthetic(n, d, k int, seed int64) ([][]float64, []int) {
	rng := rand.New(rand.NewSource(seed))
	// One random class-mean direction per class.
	means := make([][]float64, k)
	for c := range means {
		means[c] = make([]float64, d)
		for j := range means[c] {
			means[c][j] = rng.NormFloat64() * 0.6
		}
	}
	x := make([][]float64, n)
	y := make([]int, n)
	for i := range x {
		c := i % k
		y[i] = c
		row := make([]float64, d)
		for j := range row {
			row[j] = means[c][j] + rng.NormFloat64() // overlapping noise → not separable
		}
		x[i] = row
	}
	return x, y
}

// softmaxQuality returns classification accuracy and mean log-loss of a fitted
// model on (x,y) — used to confirm the Newton fit is at least as good as GD.
func softmaxQuality(t testing.TB, m *SoftmaxRegression, x [][]float64, y []int) (acc, logloss float64) {
	t.Helper()
	p, err := m.PredictProba(x)
	if err != nil {
		t.Fatal(err)
	}
	var correct int
	for i := range x {
		best, bp := 0, p[i][0]
		for c := range p[i] {
			if p[i][c] > bp {
				bp, best = p[i][c], c
			}
		}
		if best == y[i] {
			correct++
		}
		pc := p[i][y[i]]
		if pc < 1e-15 {
			pc = 1e-15
		}
		logloss += -math.Log(pc)
	}
	return float64(correct) / float64(len(x)), logloss / float64(len(x))
}

// TestSoftmaxRegressionQuality logs accuracy and log-loss on the A/B workload so
// the Newton fit can be compared against the previous GD fit (always passes).
func TestSoftmaxRegressionQuality(t *testing.T) {
	x, y := softmaxSynthetic(4000, 20, 3, 1)
	var m SoftmaxRegression
	if err := m.Fit(x, y, 3, 200, 0.05); err != nil {
		t.Fatal(err)
	}
	acc, ll := softmaxQuality(t, &m, x, y)
	t.Logf("SoftmaxRegression A/B fit: accuracy=%.6f log-loss=%.6f", acc, ll)
}

// BenchmarkSoftmaxRegressionFit measures Fit on the §V22 A/B workload
// (n=4000, d=20, 3-class), steps=200.
func BenchmarkSoftmaxRegressionFit(b *testing.B) {
	x, y := softmaxSynthetic(4000, 20, 3, 1)
	b.ResetTimer()
	for range b.N {
		var m SoftmaxRegression
		if err := m.Fit(x, y, 3, 200, 0.05); err != nil {
			b.Fatal(err)
		}
	}
}
