package nn

import (
	"math"
	"testing"
)

// TestHQQClampMatchesMathMinMax pins the rewrite the quantizer's round() depends on: the
// comparison chain must equal math.Min(maxLevel, math.Max(0, r)) BIT FOR BIT, not merely for
// ordinary values. The two cases that make a naive rewrite wrong are in the table — negative
// zero, which `r < 0` would let through where math.Max(0, -0) returns +0, and NaN, which
// compares false against both bounds and must therefore fall through unchanged.
func TestHQQClampMatchesMathMinMax(t *testing.T) {
	const maxLevel = 15
	clamp := func(r float64) float64 {
		if r <= 0 {
			r = 0
		} else if r > maxLevel {
			r = maxLevel
		}
		return r
	}
	for _, r := range []float64{
		math.Copysign(0, -1), 0, 1, 7.5, maxLevel - 1, maxLevel, maxLevel + 1, 1e9,
		-1, -1e9, math.Inf(1), math.Inf(-1), math.NaN(),
		math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
	} {
		got := clamp(r)
		want := math.Min(maxLevel, math.Max(0, r))
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("clamp(%v): got %v (bits %d), math.Min/Max gives %v (bits %d)",
				r, got, math.Float64bits(got), want, math.Float64bits(want))
		}
	}
}
