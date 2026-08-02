package classic

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// TestGBMParallelFeatureScanMatchesSerial pins the parallel feature scan to the serial one, which
// is what the work gate switches between — and the gate was just lowered fourfold, so many more
// nodes now take the parallel path.
//
// It gates the SPLIT, not the arithmetic: both arms run the same scanFeatures over the same
// features, so what could differ is only the reduction — which chunk's winner survives, and how
// ties resolve. bestSplitParallel claims to reproduce the serial ascending-feature `>` winner
// exactly, and that claim is the one being held. A tolerance would be meaningless here: the two
// paths either choose the same splits, in which case every predicted value is bit-equal, or they
// diverge at one node and the trees below it are unrelated.
//
// What it does NOT catch: flipping the reduction's tie-break from `>` to `>=`. Two chunks would
// have to produce exactly equal gains, which continuous features do not do. That mutation is
// inert here rather than uncovered, and no fixture built from real-valued data can redden it.
//
// The fixture must CROSS the gate or both arms run the same serial code: d*n at the root is
// 24*2400 = 57600, above gbmSplitParWork, and stays above it for the upper nodes that matter.
func TestGBMParallelFeatureScanMatchesSerial(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("needs more than one processor to exercise the parallel scan")
	}
	rng := rand.New(rand.NewSource(20260804))
	const n, d = 2400, 24
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := range x {
		x[i] = make([]float64, d)
		for j := range x[i] {
			x[i][j] = rng.NormFloat64()
		}
		// The signal is carried by features 7, 13 and 20 — deliberately ODD and spread across
		// different chunks of the feature split. A fixture whose best feature is always index 0
		// cannot see a chunking mistake: dropping every odd feature still leaves the winner in
		// place, and the mutation passes. Which is exactly what the first version of this test
		// did.
		y[i] = 2.0*x[i][7] + 1.3*x[i][13] + 0.9*x[i][20] + 0.05*rng.NormFloat64()
	}
	fit := func() []float64 {
		t.Helper()
		m := NewGradientBoostingRegressor(WithGBMNEstimators(6), WithGBMMaxDepth(4), WithGBMSeed(5))
		if err := m.Fit(x, y); err != nil {
			t.Fatal(err)
		}
		out, err := m.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	prev := runtime.GOMAXPROCS(1) // one worker ⇒ len(valsW) == 1 ⇒ the serial scan
	serial := fit()
	runtime.GOMAXPROCS(prev)
	par := fit()

	if len(serial) != len(par) {
		t.Fatalf("%d predictions serial, %d parallel", len(serial), len(par))
	}
	for i := range serial {
		if math.Float64bits(serial[i]) != math.Float64bits(par[i]) {
			t.Fatalf("row %d: serial %v, parallel %v — the chunked feature scan chose a different"+
				" split", i, serial[i], par[i])
		}
	}
}
