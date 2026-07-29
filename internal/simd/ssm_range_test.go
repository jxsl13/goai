package simd

import (
	"math"
	"math/rand"
	"testing"
)

// SSMScanRangeF64 over channel chunks must be BIT-IDENTICAL to the whole SSMScanF64 —
// the invariant the channel-parallel cpu kernel relies on. Covers N multiple-of-4 (all
// SIMD) and not (scalar N-tail), and several chunk sizes.
func TestSSMScanRangeF64BitExactVsWhole(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for _, dims := range [][3]int{{128, 256, 16}, {64, 128, 128}, {96, 40, 17}, {200, 64, 4}} {
		L, D, N := dims[0], dims[1], dims[2]
		mk := func(n int) []float64 {
			s := make([]float64, n)
			for i := range s {
				s[i] = rng.NormFloat64()
			}
			return s
		}
		u, delta := mk(L*D), mk(L*D)
		as := mk(D * N)
		for i := range as {
			as[i] = -0.1 - rng.Float64() // A<0 so exp arg ≤ 0
		}
		for i := range delta {
			delta[i] = 0.5 * (1 + rng.Float64()) // Δ>0
		}
		bs, cs := mk(L*N), mk(L*N)
		dsk := mk(D)
		whole := make([]float64, L*D)
		hw := make([]float64, D*N)
		SSMScanF64(u, delta, as, bs, cs, dsk, whole, hw, L, D, N)
		for _, chunk := range []int{1, 4, 7, 32} {
			got := make([]float64, L*D)
			hg := make([]float64, D*N)
			for lo := 0; lo < D; lo += chunk {
				hi := lo + chunk
				if hi > D {
					hi = D
				}
				SSMScanRangeF64(u, delta, as, bs, cs, dsk, got, hg, L, D, N, lo, hi)
			}
			for i := range got {
				if math.Float64bits(got[i]) != math.Float64bits(whole[i]) {
					t.Fatalf("L=%d D=%d N=%d chunk=%d idx=%d: chunked %v vs whole %v", L, D, N, chunk, i, got[i], whole[i])
				}
			}
		}
	}
}
