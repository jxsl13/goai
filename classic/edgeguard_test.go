package classic_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/classic"
)

// mustErrNoPanic asserts fn returns a clean error without panicking — the guard
// contract every classic estimator shares (a ragged/degenerate/before-Fit input is
// rejected, never a raw panic). Surfaced by a read-only audit; these four were the
// outliers vs their siblings (the LinearRegression-width-bug class).
func mustErrNoPanic(t *testing.T, name string, fn func() error) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s panicked (want a clean error): %v", name, r)
		}
	}()
	if err := fn(); err == nil {
		t.Errorf("%s: want an error, got nil", name)
	}
}

func TestClassicEdgeGuards(t *testing.T) {
	// KMeans: a row narrower than x[0] panicked (no ragged-X guard).
	mustErrNoPanic(t, "KMeans ragged X", func() error {
		_, _, e := classic.KMeans([][]float64{{0, 0}, {1}, {5, 5}}, [][]float64{{0, 0}, {5, 5}}, 10)
		return e
	})
	// PCA.Fit: same — a ragged row panicked.
	mustErrNoPanic(t, "PCA ragged X", func() error {
		return (&classic.PCA{}).Fit([][]float64{{1, 2, 3}, {4, 5}, {7, 8, 9}}, 2)
	})
	// SoftmaxRegression.PredictProba: dereferenced a nil W before Fit → nil panic.
	mustErrNoPanic(t, "SoftmaxRegression before Fit", func() error {
		_, e := (&classic.SoftmaxRegression{}).PredictProba([][]float64{{1, 2, 3}})
		return e
	})
}

// GaussianNB on an all-constant dataset silently produced NaN probabilities: with
// every feature globally constant, maxVar==0 so epsilon==0 and the Gaussian divides
// by zero. It must instead reject the degenerate fit (or return finite output).
func TestGaussianNBAllConstantNoNaN(t *testing.T) {
	m := classic.NewGaussianNB()
	if err := m.Fit([][]float64{{3, 3}, {3, 3}, {3, 3}, {3, 3}}, []int{0, 0, 1, 1}); err != nil {
		return // rejecting a zero-variance dataset is an acceptable outcome
	}
	p, err := m.PredictProba([][]float64{{3, 3}})
	if err != nil {
		return
	}
	for _, row := range p {
		for _, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("GaussianNB all-constant fit yielded a non-finite probability %v", v)
			}
		}
	}
}
