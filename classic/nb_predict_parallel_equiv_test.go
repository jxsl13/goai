package classic

import (
	"math/rand"
	"runtime"
	"testing"
)

// TestNBPredictClaimedRowsMatchSerial locks the block-claiming prediction loop to the serial path.
// Which worker retires a row cannot change that row's arithmetic — each writes only its own output
// and reads immutable fitted state — so the assertion is exact, on every row.
//
// GOMAXPROCS is forced to 1 for the reference, which is the serial branch the helper already has,
// and back to the machine's count for the parallel one. A row count that is NOT a multiple of the
// claim grain is included, so the final short claim is exercised.
func TestNBPredictClaimedRowsMatchSerial(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	// The sizes must clear the helper's own serial threshold of n*feat >= 8192, or the "parallel"
	// arm is never entered and the test compares the serial path against itself. A first version
	// used 13, 300 and 1027 rows at 6 features — all below it — and a mutation that SKIPPED a row
	// per claim passed. 4000 and 5003 rows at 6 features clear it, and 5003 is deliberately not a
	// multiple of the 256-row claim grain so the final short claim is exercised.
	//
	// n also cannot be 1: a single sample has zero variance in every feature and GaussianNB
	// refuses to fit it.
	for _, n := range []int{4000, 5003} {
		const d, k = 6, 3
		x := make([][]float64, n)
		y := make([]int, n)
		for i := range x {
			x[i] = make([]float64, d)
			for j := range x[i] {
				x[i][j] = rng.NormFloat64()
			}
			y[i] = rng.Intn(k)
		}
		m := NewGaussianNB()
		if err := m.Fit(x, y); err != nil {
			t.Fatal(err)
		}
		prev := runtime.GOMAXPROCS(1)
		serial, err := m.Predict(x)
		runtime.GOMAXPROCS(prev)
		if err != nil {
			t.Fatal(err)
		}
		par, err := m.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		if len(serial) != len(par) {
			t.Fatalf("n=%d: %d vs %d rows", n, len(serial), len(par))
		}
		for i := range serial {
			if serial[i] != par[i] {
				t.Fatalf("n=%d row %d: serial %v, claimed %v", n, i, serial[i], par[i])
			}
		}
	}
}
