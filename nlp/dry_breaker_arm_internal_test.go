package nlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// differs reports whether a and b hold any differing bit pattern.
func differs(a, b []float64) bool {
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return true
		}
	}
	return false
}

// TestDRYBreakerArmsAgree pins the two membership arms in applyDRY against each other.
//
// applyDRY picks its arm on len(s.DRYBreakers) alone: at or below dryBreakerScanMax it scans the
// slice, above it builds a map. The arms are supposed to compute the same predicate, so the split
// must be invisible in the output — but nothing else in the package compares them, and the existing
// DRY tests all use short breaker lists, so they only ever exercise the scan.
//
// The trick that makes the comparison exact is padding with DUPLICATES: repeating a breaker list
// changes len(DRYBreakers), and therefore the arm, while leaving the SET it denotes untouched. Any
// difference in the output is then attributable to the arm and nothing else, so this can assert
// bit-equality rather than a tolerance.
//
// Verified by mutation rather than asserted: inverting the scan's comparison reddens this, and so
// does inverting the map arm's, so BOTH paths are gated and not just the one the short list picks.
// Deleting the scan's early `break` leaves it green — that break is an exit, not a semantic, and
// this test is deliberately NOT claimed as its floor.
func TestDRYBreakerArmsAgree(t *testing.T) {
	const vocab, L = 4096, 512
	rng := rand.New(rand.NewPCG(9, 41))

	// 5 and 8 fall inside the window's value range below, so membership genuinely fires; the rest
	// never occur, which is the realistic mix (callers pass ids for punctuation the vocabulary may
	// or may not produce).
	base := []int{0, 1, 5, 8, 2}
	if len(base) > dryBreakerScanMax {
		t.Fatalf("the base breaker list (%d) no longer selects the scan arm (max %d); this test "+
			"would compare the map arm against itself", len(base), dryBreakerScanMax)
	}
	// Repeat until the list is longer than the threshold: same set, other arm.
	padded := make([]int, 0, len(base)*8)
	for len(padded) <= dryBreakerScanMax {
		padded = append(padded, base...)
	}

	run := func(breakers []int, logits []float64, hist []int) []float64 {
		out := append([]float64(nil), logits...)
		s := &Sampler{
			DRYMultiplier: 0.8,
			DRYBase:       1.75,
			DRYAllowedLen: 2,
			DRYRange:      L,
			DRYBreakers:   breakers,
		}
		s.applyDRY(out, hist)
		return out
	}

	loadBearing := 0
	for c := range 8 {
		logits := make([]float64, vocab)
		for i := range logits {
			logits[i] = rng.NormFloat64()
		}
		// A short period, so the suffix scan finds real repetitions and the penalty actually fires;
		// a purely random history over any usable alphabet almost never repeats deeply enough to
		// clear DRYAllowedLen, which an earlier version of this fixture ran into.
		period := 5 + c
		hist := make([]int, L)
		for i := range hist {
			hist[i] = (i % period) + 3 // values in [3, period+2]
		}
		for range L / 16 { // break perfect periodicity so the cases are not all one shape
			hist[rng.IntN(L)] = rng.IntN(12) + 3
		}

		scan, mapped := run(base, logits, hist), run(padded, logits, hist)
		var penalized int
		for i := range scan {
			if math.Float64bits(scan[i]) != math.Float64bits(mapped[i]) {
				t.Fatalf("case %d: logit %d is %v (%016x) via the scan arm but %v (%016x) via the "+
					"map arm — the two membership paths disagree", c, i, scan[i],
					math.Float64bits(scan[i]), mapped[i], math.Float64bits(mapped[i]))
			}
			if math.Float64bits(scan[i]) != math.Float64bits(logits[i]) {
				penalized++
			}
		}
		// Without this the case could pass by penalizing nothing at all, which would compare two
		// untouched copies of the input and prove nothing.
		if penalized == 0 {
			t.Fatalf("case %d: no logit was penalized; the fixture no longer reaches the DRY "+
				"penalty and the comparison above is vacuous", c)
		}
		// The floor above only proves the penalty ran. It does NOT prove the arms were compared on a
		// predicate that is ever TRUE: with a breaker set the window never contains, both would
		// compute all-false and agree trivially. So assert the fixture directly.
		var hits int
		for _, tok := range hist {
			for _, b := range base {
				if tok == b {
					hits++
					break
				}
			}
		}
		if hits == 0 {
			t.Fatalf("case %d: no window position is a breaker, so both arms computed a predicate "+
				"that is false everywhere and agreed for free", c)
		}
		// Stronger still, where it holds: the breakers changing the OUTPUT. This is not required of
		// every case — a breaker can sit in the window and still not alter the result, because the
		// penalty keeps only the strongest repetition per token and that one may not cross it
		// (period 8 is such a case) — but if it held for no case at all, the sweep would never
		// exercise a breaker's effect.
		if free := run(nil, logits, hist); differs(free, scan) {
			loadBearing++
		}
	}
	if loadBearing == 0 {
		t.Fatalf("no case in the sweep changed its output when the breakers were removed; the " +
			"fixture never exercises a breaker actually cutting a repetition")
	}
}
