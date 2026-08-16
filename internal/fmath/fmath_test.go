package fmath

import (
	"math"
	"testing"
)

// hostile is every value that can make the builtins and the math functions disagree, plus the
// neighbors that surround the disagreement: both NaN partners, both signed zeros, both
// extremes of the normal range and the smallest subnormal.
var hostile = []float64{
	math.NaN(), math.Inf(1), math.Inf(-1),
	0, math.Copysign(0, -1),
	1, -1, math.MaxFloat64, -math.MaxFloat64,
	math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
}

// TestMinMaxAreBitIdentical is the contract: over every ordered pair, Min and Max agree with
// math.Min and math.Max on every bit — not to a tolerance, and not modulo NaN payloads.
func TestMinMaxAreBitIdentical(t *testing.T) {
	for _, x := range hostile {
		for _, y := range hostile {
			if got, want := Min(x, y), math.Min(x, y); math.Float64bits(got) != math.Float64bits(want) {
				t.Errorf("Min(%v, %v) = %v, want %v", x, y, got, want)
			}
			if got, want := Max(x, y), math.Max(x, y); math.Float64bits(got) != math.Float64bits(want) {
				t.Errorf("Max(%v, %v) = %v, want %v", x, y, got, want)
			}
		}
	}
}

// TestTheGuardIsLoadBearing pins WHY the fallback exists by naming every pair on which the
// raw builtin would be wrong. Without this the guard looks like defensive noise and the next
// reader deletes it; the four pairs below are the entire reason it is there.
//
// It is also the floor under the test above: if the builtins ever gained math's infinity rule
// this list would empty out, and a bit-identity test that no longer discriminates would pass
// for the wrong reason.
func TestTheGuardIsLoadBearing(t *testing.T) {
	nan, pinf, ninf := math.NaN(), math.Inf(1), math.Inf(-1)
	// Keyed by BITS, not by value: a float64 map key of NaN never matches itself, so a
	// value-keyed set silently reports every pair as unlisted.
	key := func(x, y float64) [2]uint64 {
		return [2]uint64{math.Float64bits(x), math.Float64bits(y)}
	}
	want := map[[2]uint64]bool{
		key(nan, pinf): true, key(pinf, nan): true, // Max: math says +Inf, the builtin says NaN
		key(nan, ninf): true, key(ninf, nan): true, // Min: math says -Inf, the builtin says NaN
	}
	// Compare by bits, EXCEPT that any two NaNs count as agreeing. A NaN's payload is not
	// semantic, and the hardware propagates it differently per architecture: on amd64 ten further
	// pairs report "min NaN vs NaN" where the two NaNs carry different payload bits, which is a
	// fact about the CPU rather than about min/max. Keeping raw bit equality here made this guard
	// pass on arm64 and fail on amd64 with 14 divergent pairs against the 4 it documents.
	same := func(a, b float64) bool {
		if math.IsNaN(a) && math.IsNaN(b) {
			return true
		}
		return math.Float64bits(a) == math.Float64bits(b) // keeps -0 distinct from +0, which IS semantic
	}
	var found int
	for _, x := range hostile {
		for _, y := range hostile {
			rawMin, rawMax := min(x, y), max(x, y)
			minDiffers := !same(rawMin, math.Min(x, y))
			maxDiffers := !same(rawMax, math.Max(x, y))
			if !minDiffers && !maxDiffers {
				continue
			}
			found++
			if !want[key(x, y)] {
				t.Errorf("unlisted divergence at (%v, %v): min %v vs %v, max %v vs %v",
					x, y, rawMin, math.Min(x, y), rawMax, math.Max(x, y))
			}
		}
	}
	if found != len(want) {
		t.Fatalf("%d divergent pairs, want exactly %d — the guard's justification moved",
			found, len(want))
	}
}

// TestOnlyNaNResultsCanDiffer is the load-bearing property itself, stated directly: the
// builtin and math can disagree only where the BUILTIN returns NaN. That is what makes a
// NaN-triggered fallback complete rather than merely helpful, and it is the claim a reader
// has to trust when they see the guard.
func TestOnlyNaNResultsCanDiffer(t *testing.T) {
	for _, x := range hostile {
		for _, y := range hostile {
			if m := min(x, y); m == m && math.Float64bits(m) != math.Float64bits(math.Min(x, y)) {
				t.Errorf("min(%v, %v) = %v is non-NaN yet differs from math.Min", x, y, m)
			}
			if m := max(x, y); m == m && math.Float64bits(m) != math.Float64bits(math.Max(x, y)) {
				t.Errorf("max(%v, %v) = %v is non-NaN yet differs from math.Max", x, y, m)
			}
		}
	}
}

// TestSignedZeroOrdering pins the half of the contract the infinity rule distracts from. A
// comparison chain written with < instead of <= gets these wrong, which is what makes the
// builtin the safer rewrite of a clamp as well as the faster one.
func TestSignedZeroOrdering(t *testing.T) {
	nz := math.Copysign(0, -1)
	for _, c := range []struct {
		name      string
		got, want float64
	}{
		{"Min(+0,-0)", Min(0, nz), math.Min(0, nz)},
		{"Min(-0,+0)", Min(nz, 0), math.Min(nz, 0)},
		{"Max(+0,-0)", Max(0, nz), math.Max(0, nz)},
		{"Max(-0,+0)", Max(nz, 0), math.Max(nz, 0)},
	} {
		if math.Float64bits(c.got) != math.Float64bits(c.want) {
			t.Errorf("%s: signbit %v, want %v", c.name, math.Signbit(c.got), math.Signbit(c.want))
		}
	}
}
