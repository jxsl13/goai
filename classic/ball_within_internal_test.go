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

// TestBallWithinMatchesPerDimensionBailout pins the L2 bail-out rewrite against the per-dimension
// form it replaced, across every remainder class.
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
func TestBallWithinMatchesPerDimensionBailout(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 5))
	bt := &ballTree{metric: ballL2}
	var decided, bailed int
	for d := 1; d <= 9; d++ {
		for range 400 {
			a := make([]float64, d)
			b := make([]float64, d)
			for i := range a {
				a[i] = rng.NormFloat64()
				b[i] = rng.NormFloat64()
			}
			for _, eps := range []float64{0.01, 0.5, 1, 2, 5, 100} {
				want := withinReference(a, b, eps, eps*eps, false)
				got := bt.within(a, b, eps, eps*eps)
				if got != want {
					t.Fatalf("d=%d eps=%v: within=%v, per-dimension reference=%v\na=%v\nb=%v",
						d, eps, got, want, a, b)
				}
				decided++
				if !want {
					bailed++
				}
			}
		}
	}
	// Without this the sweep could pass by answering "within" every time, never exercising the
	// bail-out the rewrite touched.
	if bailed == 0 || bailed == decided {
		t.Fatalf("%d of %d cases bailed out; the sweep needs both outcomes to exercise the "+
			"threshold path", bailed, decided)
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
	bt := &ballTree{metric: ballL2}
	for _, d := range []int{1, 4, 5, 8, 20} {
		for pos := range d {
			a := make([]float64, d)
			b := make([]float64, d)
			a[pos] = math.NaN()
			const eps = 3.0
			want := withinReference(a, b, eps, eps*eps, false)
			if got := bt.within(a, b, eps, eps*eps); got != want {
				t.Fatalf("d=%d NaN at %d: within=%v, per-dimension reference=%v", d, pos, got, want)
			}
		}
	}
}

// TestBallWithinBoundaryIsInclusive pins the boundary, which the monotone argument turns on: a
// point whose squared distance is EXACTLY the threshold is within, since the test is strictly
// greater-than. A rewrite that flipped it to >= would silently shrink every neighbourhood.
func TestBallWithinBoundaryIsInclusive(t *testing.T) {
	bt := &ballTree{metric: ballL2}
	// 3-4-5 triangle in the first two dimensions, padded so the tail runs too.
	for _, d := range []int{2, 5, 6} {
		a := make([]float64, d)
		b := make([]float64, d)
		a[0], a[1] = 3, 4
		const eps = 5.0
		if !bt.within(a, b, eps, eps*eps) {
			t.Fatalf("d=%d: a point at exactly eps must be within", d)
		}
		if !withinReference(a, b, eps, eps*eps, false) {
			t.Fatalf("d=%d: the reference disagrees, so the fixture is wrong", d)
		}
	}
}
