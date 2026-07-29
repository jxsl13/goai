package simd

import (
	"math"
	"math/rand"
	"testing"
)

// WKVScanRangeF64 over 4-aligned channel chunks must be BIT-IDENTICAL to the single
// whole-range WKVScanF64 — the invariant the channel-parallel cpu kernel relies on.
// Covers d multiple-of-4 (all SIMD) and not (scalar tail), and several chunk sizes.
func TestWKVScanRangeF64BitExactVsWhole(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, dims := range [][2]int{{128, 512}, {64, 1024}, {200, 64}, {130, 17}, {96, 12}} {
		seq, d := dims[0], dims[1]
		k := make([]float64, seq*d)
		v := make([]float64, seq*d)
		w := make([]float64, d)
		u := make([]float64, d)
		for i := range k {
			k[i] = rng.NormFloat64()
			v[i] = rng.NormFloat64() * 2
		}
		for c := range w {
			w[c] = 0.5 + rng.Float64()
			u[c] = rng.NormFloat64() * 0.5
		}
		whole := make([]float64, seq*d)
		WKVScanF64(k, v, w, u, whole, seq, d)
		for _, chunk := range []int{4, 8, 16, 64} {
			got := make([]float64, seq*d)
			for lo := 0; lo < d; lo += chunk {
				hi := lo + chunk
				if hi > d {
					hi = d
				}
				WKVScanRangeF64(k, v, w, u, got, seq, d, lo, hi)
			}
			for i := range got {
				if math.Float64bits(got[i]) != math.Float64bits(whole[i]) {
					t.Fatalf("seq=%d d=%d chunk=%d idx=%d: chunked %v vs whole %v (not bit-identical)",
						seq, d, chunk, i, got[i], whole[i])
				}
			}
		}
	}
}
