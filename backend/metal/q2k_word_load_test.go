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

func TestMetalQMatMulQ2KWordLoadMatchesControl(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	previousCooperative := metal.SetQ2KCooperative(false)
	previousWord := metal.SetQ2KWordLoad(false)
	defer func() {
		metal.SetQ2KWordLoad(previousWord)
		metal.SetQ2KCooperative(previousCooperative)
	}()
	for _, tc := range []struct{ k, n int }{{256, 8}, {512, 7}, {2048, 64}, {4096, 3}, {2048, 3072}} {
		x := tensor.New(tensor.F32, tensor.Shape{1, tc.k})
		for i, value := range qmRand(tc.k, 81) {
			x.Storage().F32()[i] = float32(value)
		}
		w := tensor.FromFloat64(tensor.Shape{tc.n, tc.k}, qmRand(tc.n*tc.k, 82))
		wq, err := gguf.Quantize(w, gguf.Q2_K)
		if err != nil {
			t.Fatal(err)
		}
		if tc.k == 2048 && tc.n == 3072 {
			rowBytes := (tc.k / 256) * 84
			wq[80], wq[81] = 0x00, 0x7e
			wq[rowBytes+82], wq[rowBytes+83] = 0x00, 0x7c
		}
		xBefore := slices.Clone(x.Storage().F32())
		wBefore := slices.Clone(wq)
		metal.SetQ2KCooperative(false)
		scalar, err := metal.QMatMulQ2_K(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		metal.SetQ2KCooperative(true)
		metal.SetQ2KWordLoad(false)
		control, err := metal.QMatMulQ2_K(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		metal.SetQ2KWordLoad(true)
		candidate, err := metal.QMatMulQ2_K(x, wq, tc.n, tc.k)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(x.Storage().F32(), xBefore) || !slices.Equal(wq, wBefore) {
			t.Fatalf("k=%d n=%d: Q2_K input mutated", tc.k, tc.n)
		}
		var maxRel float64
		for i, want := range scalar.Storage().F32() {
			got, controlValue := candidate.Storage().F32()[i], control.Storage().F32()[i]
			class := func(value float32) int {
				switch {
				case math.IsNaN(float64(value)):
					return 1
				case math.IsInf(float64(value), 1):
					return 2
				case math.IsInf(float64(value), -1):
					return 3
				default:
					return 0
				}
			}
			if class(got) != class(controlValue) {
				t.Fatalf("k=%d n=%d element %d: control=%g candidate=%g class differs", tc.k, tc.n, i, controlValue, got)
			}
			if class(controlValue) != 0 {
				continue
			}
			rel := math.Abs(float64(got-want)) / math.Max(1, math.Abs(float64(want)))
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 2e-5 {
				t.Fatalf("k=%d n=%d element %d candidate=%g scalar=%g relative=%g", tc.k, tc.n, i, got, want, rel)
			}
			controlRel := math.Abs(float64(got-controlValue)) / math.Max(1, math.Abs(float64(controlValue)))
			if controlRel > 2e-5 {
				t.Fatalf("k=%d n=%d element %d candidate=%g control=%g relative=%g", tc.k, tc.n, i, got, controlValue, controlRel)
			}
		}
		t.Logf("k=%d n=%d: Q2_K word-load candidate vs scalar max relative difference %.3e", tc.k, tc.n, maxRel)
	}
}
