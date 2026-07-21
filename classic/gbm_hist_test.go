package classic

import (
	"testing"
)

// TestGBMHistogramParity fits the regressor and classifier with and without the
// histogram grower on the same data and asserts the histogram model matches the
// exact model's holdout accuracy within noise (the split thresholds become data
// quantiles, so predictions differ slightly but the boosted model is equivalent).
func TestGBMHistogramParity(t *testing.T) {
	const n, d, depth, nEst = 20000, 12, 3, 40
	xall, lab, _ := synthFitData(n+4000, d, 3)
	yf := make([]float64, len(lab)) // regression target = class id as float
	yb := make([]int, len(lab))     // binary label for the classifier
	for i, v := range lab {
		yf[i] = float64(v)
		if v >= 1 {
			yb[i] = 1
		}
	}
	xtr, ytr, yctr := xall[:n], yf[:n], yb[:n]
	xte, yte, ycte := xall[n:], yf[n:], yb[n:]

	// --- regressor: holdout MSE parity ---
	fitReg := func(hist bool) []float64 {
		opts := []GBMOption{WithGBMNEstimators(nEst), WithGBMMaxDepth(depth)}
		if hist {
			opts = append(opts, WithGBMHistogram(256))
		}
		m := NewGradientBoostingRegressor(opts...)
		if err := m.Fit(xtr, ytr); err != nil {
			t.Fatal(err)
		}
		p, err := m.Predict(xte)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	mse := func(p, y []float64) float64 {
		var s float64
		for i := range p {
			s += (p[i] - y[i]) * (p[i] - y[i])
		}
		return s / float64(len(p))
	}
	exReg, hiReg := fitReg(false), fitReg(true)
	exMSE, hiMSE := mse(exReg, yte), mse(hiReg, yte)
	t.Logf("regressor holdout MSE: exact %.5f  hist %.5f  (ratio %.4f)", exMSE, hiMSE, hiMSE/exMSE)
	if hiMSE > exMSE*1.03 {
		t.Fatalf("histogram regressor MSE %.5f > 1.03× exact %.5f", hiMSE, exMSE)
	}

	// --- classifier: holdout accuracy parity ---
	fitClf := func(hist bool) []int {
		opts := []GBMOption{WithGBMNEstimators(nEst), WithGBMMaxDepth(depth)}
		if hist {
			opts = append(opts, WithGBMHistogram(256))
		}
		m := NewGradientBoostingClassifier(opts...)
		if err := m.Fit(xtr, yctr); err != nil {
			t.Fatal(err)
		}
		p, err := m.Predict(xte)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	acc := func(p, y []int) float64 {
		c := 0
		for i := range p {
			if p[i] == y[i] {
				c++
			}
		}
		return float64(c) / float64(len(p))
	}
	exAcc, hiAcc := acc(fitClf(false), ycte), acc(fitClf(true), ycte)
	t.Logf("classifier holdout acc: exact %.4f  hist %.4f  (Δ %.4f)", exAcc, hiAcc, hiAcc-exAcc)
	if hiAcc < exAcc-0.01 {
		t.Fatalf("histogram classifier acc %.4f < exact %.4f − 0.01", hiAcc, exAcc)
	}
}

// TestGBMHistogramDeterministic asserts the histogram grower is deterministic:
// two fits on the same data give bit-identical predictions.
func TestGBMHistogramDeterministic(t *testing.T) {
	x, lab, _ := synthFitData(5000, 8, 3)
	y := make([]float64, len(lab))
	for i, v := range lab {
		y[i] = float64(v)
	}
	fit := func() []float64 {
		m := NewGradientBoostingRegressor(WithGBMNEstimators(20), WithGBMHistogram(128))
		if err := m.Fit(x, y); err != nil {
			t.Fatal(err)
		}
		p, _ := m.Predict(x[:200])
		return p
	}
	a, b := fit(), fit()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at %d: %v vs %v", i, a[i], b[i])
		}
	}
}

// BenchmarkGBMHistVsExact reports the fit speedup of the histogram grower.
func benchGBMFitMode(b *testing.B, n int, hist bool) {
	x, y, _ := synthFitData(n, 20, 3)
	yb := make([]int, len(y))
	for i, v := range y {
		if v >= 1 {
			yb[i] = 1
		}
	}
	opts := []GBMOption{WithGBMNEstimators(50)}
	if hist {
		opts = append(opts, WithGBMHistogram(256))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewGradientBoostingClassifier(opts...)
		if err := m.Fit(x, yb); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGBMHist_exact_20k(b *testing.B) { benchGBMFitMode(b, 20000, false) }
func BenchmarkGBMHist_hist_20k(b *testing.B)  { benchGBMFitMode(b, 20000, true) }
func BenchmarkGBMHist_exact_80k(b *testing.B) { benchGBMFitMode(b, 80000, false) }
func BenchmarkGBMHist_hist_80k(b *testing.B)  { benchGBMFitMode(b, 80000, true) }
