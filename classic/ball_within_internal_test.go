package classic

import (
	"math"
	"math/rand/v2"
	"testing"
)

// withinReference is the shape ballTree.within had before its bail-out was moved to every fourth
// dimension: one accumulator, checked at every dimension. It is the oracle for the tests below.
func withinReference(a, b []float64, eps, eps2 float64, l1 bool) bool {
	var s float64
	for i := range a {
		if l1 {
			s += math.Abs(a[i] - b[i])
			if s > eps {
				return false
			}
			continue
		}
		d := a[i] - b[i]
		s += d * d
		if s > eps2 {
			return false
		}
	}
	return true
}

// TestBallWithinMatchesPerDimensionBailout pins BOTH metric arms' bail-out rewrite against the
// per-dimension form each replaced, across every remainder class.
//
// The dimension sweep is the point. within now checks its threshold every FOUR dimensions and
// handles the leftovers in a scalar tail, and the only benchmark that drives it uses d=20 — exactly
// divisible by four, so it never executes that tail at all (PROC-UNROLL-TAIL-COVERAGE-001). d runs
// 1..9 here, covering all four remainders twice.
//
// eps is swept alongside, because the interesting cases are the ones where the answer is decided
// mid-vector: a threshold so large that nothing bails, so small that the first dimension bails, and
// values in between where the old code bailed at a dimension the new code has not reached yet. That
// last class is what a rewrite of a monotone early exit can get wrong, and it only appears when the
// bail-out point is not a multiple of four.
//
// Both metrics are swept. The L1 arm was converted a round after L2, once a benchmark existed to
// validate it, and it has the same monotone-accumulator argument and the same NaN tail.
func TestBallWithinMatchesPerDimensionBailout(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 5))
	var decided, bailed int
	for _, met := range []ballMetric{ballL2, ballL1} {
		bt := &ballTree{metric: met}
		l1 := met == ballL1
		sweepWithin(t, bt, l1, rng, &decided, &bailed)
	}
	// Without this the sweep could pass by answering "within" every time, never exercising the
	// bail-out the rewrite touched.
	if bailed == 0 || bailed == decided {
		t.Fatalf("%d of %d cases bailed out; the sweep needs both outcomes to exercise the "+
			"threshold path", bailed, decided)
	}
}

// sweepWithin compares within against the per-dimension reference over d = 1..9, covering every
// remainder class of the four-wide loop twice.
func sweepWithin(t *testing.T, bt *ballTree, l1 bool, rng *rand.Rand, decided, bailed *int) {
	t.Helper()
	for d := 1; d <= 9; d++ {
		for range 400 {
			a := make([]float64, d)
			b := make([]float64, d)
			for i := range a {
				a[i] = rng.NormFloat64()
				b[i] = rng.NormFloat64()
			}
			for _, eps := range []float64{0.01, 0.5, 1, 2, 5, 100} {
				want := withinReference(a, b, eps, eps*eps, l1)
				got := bt.within(a, b, eps, eps*eps)
				if got != want {
					t.Fatalf("metric=%v d=%d eps=%v: within=%v, per-dimension reference=%v\na=%v\nb=%v",
						bt.metric, d, eps, got, want, a, b)
				}
				*decided++
				if !want {
					*bailed++
				}
			}
		}
	}
}

// TestBallWithinNaNMatchesOriginal pins the one case where the obvious final comparison would have
// been wrong.
//
// With a NaN coordinate the accumulator becomes NaN. The per-dimension form never bailed, because
// NaN > eps2 is false, so it fell out of the loop and returned TRUE. Ending the rewrite with
// `s <= eps2` would return false instead — NaN compares false to everything — so it ends with
// !(s > eps2). Verified to be a real gate: with the final return written as s <= eps2 this test
// fails and the rest of the package stays green.
func TestBallWithinNaNMatchesOriginal(t *testing.T) {
	for _, met := range []ballMetric{ballL2, ballL1} {
		bt := &ballTree{metric: met}
		for _, d := range []int{1, 4, 5, 8, 20} {
			for pos := range d {
				a := make([]float64, d)
				b := make([]float64, d)
				a[pos] = math.NaN()
				const eps = 3.0
				want := withinReference(a, b, eps, eps*eps, met == ballL1)
				if got := bt.within(a, b, eps, eps*eps); got != want {
					t.Fatalf("metric=%v d=%d NaN at %d: within=%v, per-dimension reference=%v",
						met, d, pos, got, want)
				}
			}
		}
	}
}

// TestBallWithinBoundaryIsInclusive pins the boundary, which the monotone argument turns on: a
// point whose squared distance is EXACTLY the threshold is within, since the test is strictly
// greater-than. A rewrite that flipped it to >= would silently shrink every neighbourhood.
func TestBallWithinBoundaryIsInclusive(t *testing.T) {
	// L2: a 3-4-5 triangle in the first two dimensions, padded so the tail runs too.
	// L1: the same two coordinates sum to exactly 7 under Manhattan distance.
	for _, tc := range []struct {
		met ballMetric
		eps float64
	}{{ballL2, 5}, {ballL1, 7}} {
		bt := &ballTree{metric: tc.met}
		for _, d := range []int{2, 5, 6} {
			a := make([]float64, d)
			b := make([]float64, d)
			a[0], a[1] = 3, 4
			if !bt.within(a, b, tc.eps, tc.eps*tc.eps) {
				t.Fatalf("metric=%v d=%d: a point at exactly eps must be within", tc.met, d)
			}
			if !withinReference(a, b, tc.eps, tc.eps*tc.eps, tc.met == ballL1) {
				t.Fatalf("metric=%v d=%d: the reference disagrees, so the fixture is wrong", tc.met, d)
			}
		}
	}
}
