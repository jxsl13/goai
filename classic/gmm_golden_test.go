package classic

import (
	"math"
	"testing"
)

// gmmFullGolden freezes full-covariance EM's per-sample scores as raw float64 bits,
// captured before the E-step and M-step loops were parallelized.
//
// TestGMMDeterminism cannot serve this purpose: it fits twice with the SAME code and
// compares, which proves the fit is reproducible, not that it still computes what it
// used to. EM is iterative, so a change that perturbs the log-likelihood by one ulp can
// move the convergence check and land on a different iterate entirely — exactly the
// class of drift a reproducibility test cannot see.
var gmmFullGolden = [10]uint64{
	0xc02d7c06453a7406,
	0xc032ba38b7994fa3,
	0xc028a5545f9aee64,
	0xc0304a764495524e,
	0xc02b5416f4142e08,
	0xc02ea1dfd2cbd6fa,
	0xc02b331781f3cf5a,
	0xc0295de3f19648b6,
	0xc033e9c4a4d3b020,
	0xc03143809a595e34,
}

// TestGMMFullBitIdenticalToGolden holds full-covariance EM bit-for-bit against a frozen
// reference at tolerance 0. The E-step is parallel over samples and the M-step over
// components; neither may move a bit, and the per-sample log-likelihood contributions
// must still be summed in sample order for the total to match.
func TestGMMFullBitIdenticalToGolden(t *testing.T) {
	X, _ := spatialData(1200, 10)
	m := NewGaussianMixture(WithGMMComponents(4), WithGMMMaxIter(12), WithGMMSeed(7),
		WithGMMCovariance(GMMFull))
	if err := m.Fit(X); err != nil {
		t.Fatal(err)
	}
	s, err := m.ScoreSamples(X[:len(gmmFullGolden)])
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range gmmFullGolden {
		if got := math.Float64bits(s[i]); got != want {
			t.Fatalf("score %d differs: got %v (%#x) want %v (%#x) — full-covariance EM is "+
				"no longer bit-identical to its frozen reference",
				i, s[i], got, math.Float64frombits(want), want)
		}
	}
}
