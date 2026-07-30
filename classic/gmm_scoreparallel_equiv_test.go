package classic

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

// ScoreSamples/PredictProba full-cov parallel fan-out must equal the serial path
// (GOMAXPROCS=1) byte-for-byte. Fit a real full-cov GMM, then compare.
func TestGMMScoreSamplesParallelEquivSerial(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 3))
	for trial := 0; trial < 20; trial++ {
		n := 200 + rng.IntN(2000)
		d := 2 + rng.IntN(12)
		k := 2 + rng.IntN(6)
		X := make([][]float64, n)
		for i := range X {
			X[i] = make([]float64, d)
			for j := range X[i] {
				X[i][j] = rng.NormFloat64() + float64(i%k)
			}
		}
		m := NewGaussianMixture(WithGMMComponents(k), WithGMMCovariance(GMMFull), WithGMMMaxIter(8), WithGMMSeed(int64(trial+1)))
		if err := m.Fit(X); err != nil {
			continue
		}
		runtime.GOMAXPROCS(1)
		s1, _ := m.ScoreSamples(X)
		p1, _ := m.PredictProba(X)
		runtime.GOMAXPROCS(16)
		s2, _ := m.ScoreSamples(X)
		p2, _ := m.PredictProba(X)
		for i := range s1 {
			if math.Float64bits(s1[i]) != math.Float64bits(s2[i]) {
				t.Fatalf("trial %d n=%d d=%d k=%d ScoreSamples[%d]: serial %v parallel %v", trial, n, d, k, i, s1[i], s2[i])
			}
			for c := range p1[i] {
				if math.Float64bits(p1[i][c]) != math.Float64bits(p2[i][c]) {
					t.Fatalf("trial %d PredictProba[%d][%d] mismatch", trial, i, c)
				}
			}
		}
	}
	runtime.GOMAXPROCS(16)
}
