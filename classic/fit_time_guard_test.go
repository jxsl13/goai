package classic

import (
	"fmt"
	"math/rand/v2"
	"testing"
	"time"
)

// TestClassicFitTimeGuard catches order-of-magnitude fit-time regressions, which the correctness
// suite cannot see.
//
// It exists because of a real one. Swapping math.Exp for a 1.6e-7-accurate approximation in the RBF
// kernel left every test in this package passing — the models were still correct — while the SVC fit
// went from 5.8 ms to 9452 ms, because SMO stopped converging and ran to maxIter. A suite that
// checks what a model PREDICTS is blind to a change that destroys how fast it gets there.
//
// The ceilings are deliberately ~10x the times measured on an M2 Pro (DecisionTree 4.6, RandomForest
// 27.7, GradientBoosting 78, SVC_rbf 5.6, GaussianNB 0.3 ms). That is far too loose to be a
// benchmark and is not meant to be one: it is an order-of-magnitude tripwire that survives slower CI
// hardware and a loaded machine, while a 1600x regression fails it by two orders.
//
// Build tier does not affect these numbers. backend/cpu has three matmul tiers (plain Go, NEON, and
// Accelerate, the latter two behind goexperiment.simd, worth 11-18x on raw matmul), so the obvious
// question is whether the classical scorecard is measured on the slow one. It is not affected:
// SVC_rbf reads 5.84/4.88 ms on a default build against 6.76/5.53 with GOEXPERIMENT=simd — if
// anything slightly slower, and certainly not faster. These learners do not route through
// backend/cpu's gemm, so the sklearn comparison holds for any build configuration.
//
// Verified by mutation rather than assumed: injecting a 1e-7 rounding error into the RBF kernel
// (math.Round(exp(x)*1e7)/1e7) makes the SVC fit take 10.997s and this test fails with the ceiling
// message; removing it, the suite passes in 0.4s. A guard that has never been observed failing on
// the input it exists for is not a guard.
func TestClassicFitTimeGuard(t *testing.T) {
	const n, d, classes = 4000, 20, 3
	rng := rand.New(rand.NewPCG(42, 42))
	X := make([][]float64, n)
	y := make([]int, n)
	yb := make([]float64, n)
	ybi := make([]int, n)
	centers := make([][]float64, classes)
	for c := range centers {
		centers[c] = make([]float64, d)
		for j := range centers[c] {
			centers[c][j] = rng.NormFloat64() * 3
		}
	}
	for i := range X {
		c := rng.IntN(classes)
		y[i] = c
		if c != 0 {
			yb[i], ybi[i] = 1, 1
		}
		X[i] = make([]float64, d)
		for j := range X[i] {
			X[i][j] = centers[c][j] + rng.NormFloat64()
		}
	}

	cases := []struct {
		name    string
		ceiling time.Duration
		fit     func() error
	}{
		{"DecisionTree", 50 * time.Millisecond, func() error {
			return NewDecisionTreeClassifier(WithMaxDepth(12)).Fit(X, y)
		}},
		{"RandomForest100", 300 * time.Millisecond, func() error {
			return NewRandomForestClassifier(WithNumTrees(100), WithSeed(42)).Fit(X, y)
		}},
		{"GradientBoosting100", 800 * time.Millisecond, func() error {
			return NewGradientBoostingClassifier(WithGBMNEstimators(100)).Fit(X, ybi)
		}},
		{"SVC_rbf", 60 * time.Millisecond, func() error {
			return NewSVC(WithSVMKernel(SVMKernelRBF)).Fit(X, yb)
		}},
		{"GaussianNB", 10 * time.Millisecond, func() error {
			return NewGaussianNB().Fit(X, y)
		}},
	}
	for _, c := range cases {
		start := time.Now()
		if err := c.fit(); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		el := time.Since(start)
		fmt.Printf("FITGUARD %-20s %7.2f ms (ceiling %.0f ms)\n", c.name,
			float64(el.Microseconds())/1000, float64(c.ceiling.Milliseconds()))
		if el > c.ceiling {
			t.Errorf("%s fit took %v, over the %v order-of-magnitude ceiling — a regression this "+
				"large is a convergence or algorithmic failure, not machine noise", c.name, el, c.ceiling)
		}
	}
}
