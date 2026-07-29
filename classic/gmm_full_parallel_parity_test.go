package classic_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/classic"
)

// PredictProba's row scan now takes the parallel path for FULL covariance, which it previously
// refused because the density kernel's solve buffers lived on the receiver. The parallel result
// must be bit-identical to the serial one.
//
// The two paths are selected by a work threshold (len(x)*k), so the test drives them by SIZE:
// a batch above the threshold parallelizes, and the same rows fed one at a time fall below it
// and stay serial. That makes the comparison exercise the real gate rather than a flag.
func TestGMMFullPredictProbaParallelMatchesSerial(t *testing.T) {
	const n, k, d = 512, 8, 16
	rng := rand.New(rand.NewSource(3))
	x := make([][]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		x[i] = row
	}
	m := classic.NewGaussianMixture(classic.WithGMMComponents(k), classic.WithGMMSeed(1),
		classic.WithGMMCovariance(classic.GMMFull))
	if err := m.Fit(x); err != nil {
		t.Fatal(err)
	}
	parallel, err := m.PredictProba(x) // above the threshold
	if err != nil {
		t.Fatal(err)
	}
	for i := range x {
		serial, err := m.PredictProba(x[i : i+1]) // one row: below the threshold
		if err != nil {
			t.Fatal(err)
		}
		for c := range k {
			if math.Float64bits(parallel[i][c]) != math.Float64bits(serial[0][c]) {
				t.Fatalf("row %d component %d: parallel %016x, serial %016x — chunking changed a value, "+
					"which it must not: every worker owns its own lr and solve buffers",
					i, c, math.Float64bits(parallel[i][c]), math.Float64bits(serial[0][c]))
			}
		}
	}
}

// k NOT divisible by 4, which is the case the first version of this suite missed entirely.
//
// logGaussianFullBatch unroll-and-jams by four and finishes the remainder by calling the
// scalar logGaussian. That tail read its forward-substitution buffer off the receiver, so once
// the full-covariance row scan went parallel it raced — but only for k%4 != 0, and every
// benchmark and test here used k=8. The race detector cannot flag a line that never executes.
//
// Runs both the parity check and, under -race, the detector over a genuinely concurrent scan.
func TestGMMFullPredictProbaTailComponentCount(t *testing.T) {
	for _, k := range []int{5, 6, 7} { // 1, 2 and 3 components in the tail
		const n, d = 512, 12
		rng := rand.New(rand.NewSource(int64(k)))
		x := make([][]float64, n)
		for i := range x {
			row := make([]float64, d)
			for j := range row {
				row[j] = rng.NormFloat64()
			}
			x[i] = row
		}
		m := classic.NewGaussianMixture(classic.WithGMMComponents(k), classic.WithGMMSeed(1),
			classic.WithGMMCovariance(classic.GMMFull))
		if err := m.Fit(x); err != nil {
			t.Fatal(err)
		}
		parallel, err := m.PredictProba(x) // above the work threshold
		if err != nil {
			t.Fatal(err)
		}
		for i := range x {
			serial, err := m.PredictProba(x[i : i+1]) // below it
			if err != nil {
				t.Fatal(err)
			}
			for c := range k {
				if math.Float64bits(parallel[i][c]) != math.Float64bits(serial[0][c]) {
					t.Fatalf("k=%d row %d component %d: parallel %016x, serial %016x — the k%%4 tail "+
						"disagrees, which is where the shared scratch was",
						k, i, c, math.Float64bits(parallel[i][c]), math.Float64bits(serial[0][c]))
				}
			}
		}
	}
}
