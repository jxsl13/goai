//go:build arm64 && goexperiment.simd

package simd

import (
	"math"
	"math/rand"
	"testing"
)

func TestSSMScanF64UnsafeDomainFallsBackBeforeMutation(t *testing.T) {
	cases := []struct {
		name  string
		delta float64
		a     float64
	}{
		{name: "negative-delta", delta: -0.25, a: -1},
		{name: "positive-a", delta: 0.25, a: 1},
		{name: "deep-underflow", delta: 709, a: -1},
		{name: "nan-delta", delta: math.NaN(), a: -1},
		{name: "inf-a", delta: 0.25, a: math.Inf(-1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const L, D, N = 3, 2, 4
			rng := rand.New(rand.NewSource(91))
			u, delta, as, bs, cs, dsk := ssmInputs(rng, L, D, N, true)
			delta[1] = tc.delta
			as[N] = tc.a
			got := make([]float64, L*D)
			want := make([]float64, L*D)
			hGot := make([]float64, D*N)
			hWant := make([]float64, D*N)
			for i := range hGot {
				hGot[i] = float64(i+1) / 17
				hWant[i] = hGot[i]
			}
			ssmScanScalar(u, delta, as, bs, cs, dsk, want, hWant, L, D, N, 0, N)
			SSMScanF64(u, delta, as, bs, cs, dsk, got, hGot, L, D, N)
			for i := range got {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
					t.Fatalf("output %d: got %v want %v", i, got[i], want[i])
				}
			}
			for i := range hGot {
				if math.Float64bits(hGot[i]) != math.Float64bits(hWant[i]) {
					t.Fatalf("state %d: got %v want %v", i, hGot[i], hWant[i])
				}
			}
		})
	}
}
