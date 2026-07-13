package nn

import (
	"math"
	"testing"
)

// §V16 tier-1 (shrink_lp proximal operator): sign(x)·relu(|x| − (1/β)|x|^(p−1)); for p=1 the
// ordinary soft-threshold; 0 maps to 0.
func TestShrinkLp(t *testing.T) {
	// p=1 soft-threshold: shrink(x,β) = sign(x)·max(|x|−1/β, 0).
	for _, c := range []struct{ x, beta, want float64 }{
		{2.0, 1.0, 1.0},   // 2 − 1 = 1
		{0.5, 1.0, 0.0},   // |0.5| ≤ 1/β ⇒ 0
		{-3.0, 2.0, -2.5}, // −(3 − 0.5)
		{0, 5, 0},
	} {
		if got := shrinkLp(c.x, c.beta, 1); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("shrinkLp(%g,%g,1)=%g want %g", c.x, c.beta, got, c.want)
		}
	}
	// p<1 (p=0.7): threshold is (1/β)|x|^(p−1); verify the formula directly.
	const p, beta = 0.7, 2.0
	for _, x := range []float64{0.3, 1.0, -2.5, 4.0} {
		a := math.Abs(x)
		thr := (1 / beta) * math.Pow(a, p-1)
		want := 0.0
		if a > thr {
			want = math.Copysign(a-thr, x)
		}
		if got := shrinkLp(x, beta, p); math.Abs(got-want) > 1e-12 {
			t.Errorf("shrinkLp(%g,%g,%g)=%g want %g", x, beta, p, got, want)
		}
	}
}
