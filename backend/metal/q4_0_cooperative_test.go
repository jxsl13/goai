//go:build darwin && cgo

package metal_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// TestMetalQMatMulQ4_0CooperativeMatchesScalar mirrors the Q4_K/Q6_K cooperative
// parity tests: the cooperative kernel keeps the scalar kernel's per-element
// arithmetic but reduces per-lane partials with simd_sum, so it is held to the same
// 2e-5 relative bar rather than to bit-identity. Odd N exercises the tail row where
// a threadgroup's second simdgroup has no work.
func TestMetalQMatMulQ4_0CooperativeMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	for _, tc := range []struct{ k, n int }{{2048, 7}, {256, 8}, {512, 1}, {4096, 3}, {2048, 64}} {
		x := tensor.New(tensor.F32, tensor.Shape{1, tc.k})
		for i, value := range qmRand(tc.k, 31) {
			x.Storage().F32()[i] = float32(value)
		}
		w := tensor.FromFloat64(tensor.Shape{tc.n, tc.k}, qmRand(tc.n*tc.k, 32))
		wq, err := gguf.Quantize(w, gguf.Q4_0)
		if err != nil {
			t.Fatal(err)
		}
		previous := metal.SetQ4_0Cooperative(false)
		scalar, err := metal.QMatMulQ4_0(x, wq, tc.n, tc.k)
		if err != nil {
			metal.SetQ4_0Cooperative(previous)
			t.Fatal(err)
		}
		metal.SetQ4_0Cooperative(true)
		cooperative, err := metal.QMatMulQ4_0(x, wq, tc.n, tc.k)
		metal.SetQ4_0Cooperative(previous)
		if err != nil {
			t.Fatal(err)
		}
		var maxRel float64
		for i, want := range scalar.Storage().F32() {
			got := cooperative.Storage().F32()[i]
			den := math.Abs(float64(want))
			if den < 1 {
				den = 1
			}
			rel := math.Abs(float64(got-want)) / den
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 2e-5 {
				t.Fatalf("k=%d n=%d element %d cooperative=%g scalar=%g relative=%g", tc.k, tc.n, i, got, want, rel)
			}
		}
		t.Logf("k=%d n=%d: Q4_0 cooperative vs scalar max relative difference %.3e", tc.k, tc.n, maxRel)
	}
}
