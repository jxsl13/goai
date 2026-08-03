package classic

import (
	"math/rand"
	"runtime"
	"testing"
)

// TestClaimedPredictionsMatchSerial locks the block-claiming prediction loops of the forest and the
// SVC to their serial paths. Which worker retires a row cannot change that row's arithmetic — each
// writes only its own output and reads immutable fitted state — so the assertion is exact.
//
// The fixtures are sized past each helper's OWN serial threshold
// (§A-PARALLEL-GATE-MUST-CLEAR-THE-SERIAL-THRESHOLD): the forest goes serial below
// rows*trees = 8192 and the SVC below rows*supportVectors = 4096, so a smaller fixture would
// compare the serial path against itself and see nothing.
func TestClaimedPredictionsMatchSerial(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	const n, d = 900, 5 // 900 rows x 16 trees = 14400, past the forest's 8192
	x := make([][]float64, n)
	y := make([]int, n)
	for i := range x {
		x[i] = make([]float64, d)
		for j := range x[i] {
			x[i][j] = rng.NormFloat64()
		}
		y[i] = rng.Intn(3)
	}
	both := func(name string, predict func() []float64) {
		t.Helper()
		prev := runtime.GOMAXPROCS(1)
		serial := predict()
		runtime.GOMAXPROCS(prev)
		par := predict()
		if len(serial) != len(par) {
			t.Fatalf("%s: %d vs %d rows", name, len(serial), len(par))
		}
		for i := range serial {
			if serial[i] != par[i] {
				t.Fatalf("%s row %d: serial %v, claimed %v", name, i, serial[i], par[i])
			}
		}
	}

	rf := NewRandomForestClassifier(WithNumTrees(16), WithSeed(3))
	if err := rf.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	both("forest", func() []float64 {
		out, err := rf.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		f := make([]float64, len(out))
		for i, v := range out {
			f[i] = float64(v)
		}
		return f
	})

	yb := make([]float64, n) // SVC is binary and takes float labels
	for i := range yb {
		yb[i] = float64(y[i] % 2)
	}
	sv := NewSVC()
	if err := sv.Fit(x, yb); err != nil {
		t.Fatal(err)
	}
	both("svc", func() []float64 {
		out, err := sv.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		f := make([]float64, len(out))
		for i, v := range out {
			f[i] = float64(v)
		}
		return f
	})
}
