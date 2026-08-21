//go:build arm64 && goexperiment.simd

package simd

import (
	"math"
	"testing"
)

func TestWKVNeonFreshSentinelAndDeepUnderflow(t *testing.T) {
	k := []float64{0, 0, 0.25, -0.25}
	v := []float64{1.5, -2.5, 3, 4}
	w := []float64{0.75, 1.25}
	u := []float64{0, 0}
	if !wkvNeonPairSafe(k, v, w, u, nil, nil, nil, 2, 2, 0) {
		t.Fatal("fresh PP=-1e38 sentinel must remain on the fused path")
	}
	out := make([]float64, len(k))
	WKVScanF64(k, v, w, u, out, 2, 2)
	for c := range 2 {
		if math.Float64bits(out[c]) != math.Float64bits(v[c]) {
			t.Fatalf("channel %d first output %x != value %x; deep-underflow exp must be +0.0",
				c, math.Float64bits(out[c]), math.Float64bits(v[c]))
		}
	}
}

func TestWKVNeonUnsafePairsFallBackBeforeMutation(t *testing.T) {
	tests := []struct {
		name     string
		unsafeLo int
		mutate   func(k, v, w, u, aa, bb, pp []float64)
	}{
		{"nan-k", 2, func(k, _, _, _, _, _, _ []float64) { k[2] = math.NaN() }},
		{"inf-v", 2, func(_, v, _, _, _, _, _ []float64) { v[3] = math.Inf(1) }},
		{"nan-w", 0, func(_, _, w, _, _, _, _ []float64) { w[0] = math.NaN() }},
		{"inf-u", 0, func(_, _, _, u, _, _, _ []float64) { u[1] = math.Inf(-1) }},
		{"nan-aa", 0, func(_, _, _, _, aa, _, _ []float64) { aa[0] = math.NaN() }},
		{"overflow-ppw", 0, func(_, _, w, _, _, _, pp []float64) {
			w[0] = -math.MaxFloat64
			pp[0] = math.MaxFloat64
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			const seq, d = 4, 4
			k := []float64{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6}
			v := []float64{1, 2, 3, 4, -1, -2, -3, -4, 0.5, 0.25, -0.5, -0.25, 2, -2, 1.5, -1.5}
			w := []float64{0.5, 0.75, 1, 1.25}
			u := []float64{0.1, -0.1, 0.2, -0.2}
			aa := []float64{0.25, -0.5, 0.75, -1}
			bb := []float64{1, 1.25, 1.5, 1.75}
			pp := []float64{0.2, -0.3, 0.4, -0.5}
			tc.mutate(k, v, w, u, aa, bb, pp)

			wantOut := make([]float64, seq*d)
			wantAA := append([]float64(nil), aa...)
			wantBB := append([]float64(nil), bb...)
			wantPP := append([]float64(nil), pp...)
			wkvScanStateScalar(k, v, w, u, wantOut, wantAA, wantBB, wantPP, seq, d, 0, d)

			gotOut := make([]float64, seq*d)
			gotAA := append([]float64(nil), aa...)
			gotBB := append([]float64(nil), bb...)
			gotPP := append([]float64(nil), pp...)
			WKVScanStateF64(k, v, w, u, gotOut, gotAA, gotBB, gotPP, seq, d)

			closeEnough := func(got, want float64) bool {
				if math.Float64bits(got) == math.Float64bits(want) || math.IsNaN(got) && math.IsNaN(want) {
					return true
				}
				return math.Abs(got-want) <= 1e-10*math.Max(1e-6, math.Abs(want))
			}
			for i := range wantOut {
				c := i % d
				unsafe := c >= tc.unsafeLo && c < tc.unsafeLo+2
				if unsafe && math.Float64bits(gotOut[i]) != math.Float64bits(wantOut[i]) {
					t.Fatalf("output[%d] bits %x != scalar %x", i, math.Float64bits(gotOut[i]), math.Float64bits(wantOut[i]))
				}
				if !unsafe && !closeEnough(gotOut[i], wantOut[i]) {
					t.Fatalf("safe output[%d] %.17g exceeds scalar tolerance %.17g", i, gotOut[i], wantOut[i])
				}
			}
			for i := range wantAA {
				unsafe := i >= tc.unsafeLo && i < tc.unsafeLo+2
				if unsafe && (math.Float64bits(gotAA[i]) != math.Float64bits(wantAA[i]) ||
					math.Float64bits(gotBB[i]) != math.Float64bits(wantBB[i]) ||
					math.Float64bits(gotPP[i]) != math.Float64bits(wantPP[i])) {
					t.Fatalf("state[%d] differs bitwise from scalar fallback", i)
				}
				if !unsafe && (!closeEnough(gotAA[i], wantAA[i]) || !closeEnough(gotBB[i], wantBB[i]) || !closeEnough(gotPP[i], wantPP[i])) {
					t.Fatalf("safe state[%d] exceeds scalar tolerance", i)
				}
			}
		})
	}
}
