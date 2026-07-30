package nlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestAggregatePrefixAttnBitIdentical pins the four-rows-per-pass aggregate against the
// one-row-at-a-time form it replaced, at tolerance ZERO.
//
// IT SWEEPS THE HEAD COUNT BECAUSE THE EXISTING TESTS DO NOT. Every SnapKV test in the package
// uses 2 or 3 observation rows, all below the group of 4, so they exercise only the tail loop
// and would pass unchanged no matter what the unrolled body computed. The sweep runs 0 through
// 9 heads so that every residue mod 4 is covered — both a body-only count (4, 8), a tail-only
// count (1, 2, 3) and every mixed count.
//
// The reference is the original loop transcribed verbatim, not rewritten: these are pure adds
// with no multiply, so there is no FMA contraction to differ between shapes, but transcribing
// keeps the ascending-head order explicit, which is the property the unroll must preserve.
func TestAggregatePrefixAttnBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	for heads := 0; heads <= 9; heads++ {
		for _, winStart := range []int{0, 1, 3, 8, 17, 64} {
			obs := make([][]float64, heads)
			for h := range obs {
				obs[h] = make([]float64, winStart)
				for j := range obs[h] {
					// A wide exponent spread, so any reassociation of the sum shows up as a
					// changed result rather than cancelling: small terms are lost when added
					// to a large running total but survive when grouped with each other.
					obs[h][j] = rng.NormFloat64() * math.Pow(2, float64(rng.IntN(40)-20))
				}
			}
			want := make([]float64, winStart)
			for _, row := range obs {
				for j := range winStart {
					want[j] += row[j]
				}
			}
			got := aggregatePrefixAttn(obs, winStart)
			if len(got) != len(want) {
				t.Fatalf("heads=%d winStart=%d: %d elements, want %d", heads, winStart, len(got), len(want))
			}
			for j := range want {
				if math.Float64bits(got[j]) != math.Float64bits(want[j]) {
					t.Fatalf("heads=%d winStart=%d element %d: %016x != %016x — the four-row group "+
						"changed the accumulation order", heads, winStart, j,
						math.Float64bits(got[j]), math.Float64bits(want[j]))
				}
			}
		}
	}
}
