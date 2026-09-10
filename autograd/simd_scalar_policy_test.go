package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// f64ScalarPolicyEqual is only for comparisons with deliberately approximate
// ARM64 SIMD leaves. Exact CPU/reference and same-SIMD checks do not use it.
func f64ScalarPolicyEqual(got, want, floor, tolerance float64) bool {
	if math.Float64bits(got) == math.Float64bits(want) {
		return true
	}
	if math.IsNaN(got) || math.IsNaN(want) || math.IsInf(got, 0) || math.IsInf(want, 0) {
		return false
	}
	if got == 0 && want == 0 {
		return false // Equal zero signs were handled by the raw-bit comparison.
	}
	return math.Abs(got-want)/math.Max(floor, math.Abs(want)) <= tolerance
}

func requireScalarParityLayout(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatal("scalar parity requires two nonnil tensors")
	}
	if got.Dtype() != want.Dtype() || !got.Shape().Equal(want.Shape()) || got.Numel() != want.Numel() {
		t.Fatalf("scalar parity layout: got %v %v (%d), want %v %v (%d)",
			got.Dtype(), got.Shape(), got.Numel(), want.Dtype(), want.Shape(), want.Numel())
	}
	var gn, wn int
	switch got.Dtype() {
	case tensor.F32:
		gn, wn = len(got.Storage().F32()), len(want.Storage().F32())
	case tensor.F64:
		gn, wn = len(got.Storage().F64()), len(want.Storage().F64())
	default:
		t.Fatalf("scalar parity requires F32 or F64, got %v", got.Dtype())
	}
	if gn != got.Numel() || wn != want.Numel() || !got.IsContiguous() || !want.IsContiguous() {
		t.Fatalf("scalar parity requires dense storage: got %d, want %d, numel %d", gn, wn, got.Numel())
	}
}

func TestF64ScalarPolicyEqual(t *testing.T) {
	negZero := math.Copysign(0, -1)
	nan := math.Float64frombits(0x7ff8000000000042)
	for _, tc := range []struct {
		name                        string
		got, want, floor, tolerance float64
		equal                       bool
	}{
		{"equal-positive", 2, 2, 1, 1e-13, true},
		{"equal-negative", -2, -2, 1, 1e-13, true},
		{"equal-positive-zero", 0, 0, 1, 1e-13, true},
		{"equal-negative-zero", negZero, negZero, 1, 1e-13, true},
		{"opposite-zero-signs", 0, negZero, 1, 1e-13, false},
		{"opposite-zero-signs-reverse", negZero, 0, 1, 1e-13, false},
		{"equal-nan-payload", nan, nan, 1, 1e-13, true},
		{"different-nan-payload", math.Float64frombits(0x7ff8000000000084), nan, 1, 1e-13, false},
		{"different-nan-sign", math.Float64frombits(0xfff8000000000042), nan, 1, 1e-13, false},
		{"nan-finite", nan, 1, 1, 1e-13, false},
		{"finite-nan", 1, nan, 1, 1e-13, false},
		{"equal-positive-infinity", math.Inf(1), math.Inf(1), 1, 1e-13, true},
		{"equal-negative-infinity", math.Inf(-1), math.Inf(-1), 1, 1e-13, true},
		{"opposite-infinities", math.Inf(1), math.Inf(-1), 1, 1e-13, false},
		{"opposite-infinities-reverse", math.Inf(-1), math.Inf(1), 1, 1e-13, false},
		{"positive-infinity-finite", math.Inf(1), math.MaxFloat64, 1, 1e-13, false},
		{"finite-positive-infinity", math.MaxFloat64, math.Inf(1), 1, 1e-13, false},
		{"negative-infinity-finite", math.Inf(-1), -math.MaxFloat64, 1, 1e-13, false},
		{"finite-negative-infinity", -math.MaxFloat64, math.Inf(-1), 1, 1e-13, false},
		{"one-ulp", math.Nextafter(1, 2), 1, 1, 1e-13, true},
		{"focal-boundary", 1e-13, 0, 1, 1e-13, true},
		{"focal-inside", math.Nextafter(1e-13, 0), 0, 1, 1e-13, true},
		{"focal-outside", math.Nextafter(1e-13, math.Inf(1)), 0, 1, 1e-13, false},
		{"focal-negative-inside", -0.5e-13, 0, 1, 1e-13, true},
		{"focal-negative-outside", -2e-13, 0, 1, 1e-13, false},
		{"focal-relative-inside", 1000 + 0.5e-10, 1000, 1, 1e-13, true},
		{"focal-relative-outside", 1000 + 2e-10, 1000, 1, 1e-13, false},
		{"wkv-floor-inside", 0.5e-16, 0, 1e-6, 1e-10, true},
		{"wkv-floor-outside", 2e-16, 0, 1e-6, 1e-10, false},
		{"wkv-negative-floor-inside", -0.5e-16, 0, 1e-6, 1e-10, true},
		{"wkv-negative-floor-outside", -2e-16, 0, 1e-6, 1e-10, false},
		{"wkv-relative-inside", -1 + 0.5e-10, -1, 1e-6, 1e-10, true},
		{"wkv-relative-outside", -1 + 2e-10, -1, 1e-6, 1e-10, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if equal := f64ScalarPolicyEqual(tc.got, tc.want, tc.floor, tc.tolerance); equal != tc.equal {
				t.Fatalf("got=%016x want=%016x floor=%g tolerance=%g: equal=%v, want %v",
					math.Float64bits(tc.got), math.Float64bits(tc.want), tc.floor, tc.tolerance, equal, tc.equal)
			}
		})
	}
}
