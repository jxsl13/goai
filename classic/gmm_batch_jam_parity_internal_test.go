package classic

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestLogGaussianFullBatchMatchesScalar is the oracle the jammed full-covariance kernel never had:
// logGaussianFullBatch against k independent logGaussian calls, bit-for-bit.
//
// The jam claims to reproduce the per-component scalar path exactly, so THAT is the reference. The
// nearest existing test, the k=5/6/7 parity check, compares the PARALLEL row scan against the
// SERIAL one — and both arms call the same batch kernel, so a defect inside the jam moves them
// together and the comparison stays green. Verified rather than assumed: perturbing the 2-wide jam
// by one ulp left the entire classic suite passing before this test existed.
//
// k IS SWEPT ACROSS EVERY REMAINDER MOD 4, because the kernel dispatches by remainder: 4 takes the
// 4-jam alone, 6 takes 4-jam plus 2-jam, 7 takes 4-jam plus 2-jam plus one scalar, 5 takes 4-jam
// plus one scalar, and 2 and 3 skip the 4-jam entirely. A fixture at a single k exercises one of
// those five paths and reads as if it covered the kernel.
func TestLogGaussianFullBatchMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 29))
	for _, k := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 11} {
		for _, d := range []int{1, 3, 12, 24} {
			m := fitFullForJamTest(t, rng, k, d)
			x := make([]float64, d)
			for i := range x {
				x[i] = rng.NormFloat64()
			}
			// Batch arm: one call filling every component.
			ld := make([]float64, k)
			var y4 [4][]float64
			for i := range y4 {
				y4[i] = make([]float64, d)
			}
			y := make([]float64, d)
			m.logGaussianFullBatch(x, ld, y4, y)

			// Reference arm: k independent scalar calls, each with its own scratch so the batch
			// kernel's buffer reuse cannot leak into the reference.
			for c := range k {
				yr := make([]float64, d)
				want, err := m.logGaussian(x, c, yr)
				if err != nil {
					t.Fatalf("k=%d d=%d c=%d: %v", k, d, c, err)
				}
				if math.Float64bits(ld[c]) != math.Float64bits(want) {
					t.Fatalf("k=%d d=%d component %d: batch %v (%016x) != scalar %v (%016x) — the "+
						"jam does not reproduce the per-component path it replaced",
						k, d, c, ld[c], math.Float64bits(ld[c]), want, math.Float64bits(want))
				}
			}
		}
	}
}

// fitFullForJamTest returns a fitted full-covariance mixture, which is the only way to get valid
// chol/invCholDiag/logDetHalf without duplicating the factorization here — a hand-built one would
// be a second implementation to keep in step.
func fitFullForJamTest(t *testing.T, rng *rand.Rand, k, d int) *GaussianMixture {
	t.Helper()
	n := k * 24
	x := make([][]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			// Cluster structure so the components separate and the covariances stay well
			// conditioned; a degenerate fit would error out rather than test the kernel.
			row[j] = rng.NormFloat64() + float64(i%k)*3
		}
		x[i] = row
	}
	m := NewGaussianMixture(WithGMMComponents(k), WithGMMCovariance(GMMFull),
		WithGMMSeed(int64(k*100+d)), WithGMMMaxIter(5))
	if err := m.Fit(x); err != nil {
		t.Skipf("k=%d d=%d: degenerate fit (%v)", k, d, err)
	}
	return m
}
