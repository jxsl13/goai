//go:build darwin && cgo

package metal_test

import (
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// TestMetalQMatMulQ5KCooperativeMatchesScalar holds both the historical
// cooperative control and the aligned-vector-load candidate to the scalar
// kernel's 2e-5 relative bar. Odd N exercises the tail row where a
// threadgroup's second simdgroup has no work.
func TestMetalQMatMulQ5KCooperativeMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	previousCooperative := metal.SetQ5KCooperative(false)
	previousWideLoad := metal.SetQ5KWideLoad(false)
	defer func() {
		metal.SetQ5KWideLoad(previousWideLoad)
		metal.SetQ5KCooperative(previousCooperative)
	}()
	for _, tc := range []struct{ k, n int }{{2048, 7}, {256, 8}, {512, 1}, {4096, 3}, {2048, 64}, {2048, 3072}} {
		x := tensor.New(tensor.F32, tensor.Shape{1, tc.k})
		for i, value := range qmRand(tc.k, 31) {
			x.Storage().F32()[i] = float32(value)
		}
		w := tensor.FromFloat64(tensor.Shape{tc.n, tc.k}, qmRand(tc.n*tc.k, 32))
		wq, err := gguf.Quantize(w, gguf.Q5_K)
		if err != nil {
			t.Fatal(err)
		}
		if tc.k == 2048 && tc.n == 3072 {
			// Q5_K stores d as f16 in the first two bytes. Plant a quiet NaN
			// in the candidate-eligible cell so all three routes must preserve
			// the quant format's non-finite classification.
			wq[0], wq[1] = 0x00, 0x7e
		}
		xBefore := slices.Clone(x.Storage().F32())
		wBefore := slices.Clone(wq)
		metal.SetQ5KCooperative(false)
		scalar, err := metal.QMatMulQ5_K(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		metal.SetQ5KCooperative(true)
		metal.SetQ5KWideLoad(false)
		control, err := metal.QMatMulQ5_K(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		metal.SetQ5KWideLoad(true)
		candidate, err := metal.QMatMulQ5_K(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(x.Storage().F32(), xBefore) || !slices.Equal(wq, wBefore) {
			t.Fatalf("k=%d n=%d: Q5_K input mutated", tc.k, tc.n)
		}
		var maxRel float64
		for i, want := range scalar.Storage().F32() {
			got := candidate.Storage().F32()[i]
			controlValue := control.Storage().F32()[i]
			gotNaN := math.IsNaN(float64(got))
			if gotNaN != math.IsNaN(float64(want)) || gotNaN != math.IsNaN(float64(controlValue)) {
				t.Fatalf("k=%d n=%d element %d: scalar/control/candidate NaN class differs", tc.k, tc.n, i)
			}
			if gotNaN {
				continue
			}
			den := math.Abs(float64(want))
			if den < 1 {
				den = 1
			}
			rel := math.Abs(float64(got-want)) / den
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 2e-5 {
				t.Fatalf("k=%d n=%d element %d candidate=%g scalar=%g relative=%g", tc.k, tc.n, i, got, want, rel)
			}
			controlDen := math.Abs(float64(controlValue))
			if controlDen < 1 {
				controlDen = 1
			}
			controlRel := math.Abs(float64(got-controlValue)) / controlDen
			if controlRel > 2e-5 {
				t.Fatalf("k=%d n=%d element %d candidate=%g control=%g relative=%g", tc.k, tc.n, i, got, controlValue, controlRel)
			}
		}
		t.Logf("k=%d n=%d: Q5_K vector-load candidate vs scalar max relative difference %.3e", tc.k, tc.n, maxRel)
	}
}
