package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func norm(t *tensor.Tensor) float64 {
	var s float64
	for i := range t.Numel() {
		v := t.AtF64(tensor.Unravel(i, t.Shape())...)
		s += v * v
	}
	return math.Sqrt(s)
}

func dot(a, b *tensor.Tensor) float64 {
	var s float64
	for i := range a.Numel() {
		c := tensor.Unravel(i, a.Shape())
		s += a.AtF64(c...) * b.AtF64(c...)
	}
	return s
}

// §V16 tier-1 (Shoemake 1985, endpoints): SLERP(t=0)=a, SLERP(t=1)=b.
func TestSLERPEndpoints(t *testing.T) {
	a := toT([][]float64{{1, 2}, {3, 4}})
	b := toT([][]float64{{-1, 0.5}, {2, -3}})
	for _, tc := range []struct {
		tt   float64
		want *tensor.Tensor
	}{{0, a}, {1, b}} {
		got, err := nn.SLERP(a, b, tc.tt)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got.Numel() {
			c := tensor.Unravel(i, got.Shape())
			if math.Abs(got.AtF64(c...)-tc.want.AtF64(c...)) > 1e-9 {
				t.Errorf("t=%g at %v: %g != %g", tc.tt, c, got.AtF64(c...), tc.want.AtF64(c...))
			}
		}
	}
}

// §V16 tier-1 (great-circle / norm preservation): with equal-norm inputs the interpolated
// tensor keeps that norm for every t — the defining property vs linear interpolation, which
// sags toward the origin at the midpoint.
func TestSLERPNormPreserved(t *testing.T) {
	// two orthogonal unit-norm vectors (Ω=90°).
	a := toT([][]float64{{1, 0, 0, 0}})
	b := toT([][]float64{{0, 1, 0, 0}})
	for _, tt := range []float64{0.1, 0.25, 0.5, 0.75, 0.9} {
		got, _ := nn.SLERP(a, b, tt)
		if math.Abs(norm(got)-1) > 1e-9 {
			t.Errorf("t=%g: SLERP norm %g, want 1 (great circle)", tt, norm(got))
		}
	}
	// midpoint of LERP would sag: |0.5(a+b)| = √0.5 ≈ 0.707 < 1.
	lerpMid := math.Sqrt(0.5)
	if lerpMid > 0.99 {
		t.Fatal("sanity: LERP midpoint should sag below 1")
	}
}

// §V16 tier-1 (constant angular velocity): the t=0.5 point bisects the angle — it is
// equidistant in angle from a and from b.
func TestSLERPMidpointBisectsAngle(t *testing.T) {
	a := toT([][]float64{{1, 0, 0}})
	b := toT([][]float64{{0.5, math.Sqrt(3) / 2, 0}}) // 60° from a, both unit norm
	mid, _ := nn.SLERP(a, b, 0.5)
	angAM := math.Acos(dot(a, mid) / (norm(a) * norm(mid)))
	angMB := math.Acos(dot(mid, b) / (norm(mid) * norm(b)))
	if math.Abs(angAM-angMB) > 1e-9 {
		t.Errorf("midpoint not angle-bisecting: ∠(a,mid)=%g vs ∠(mid,b)=%g", angAM, angMB)
	}
}

// §V16 tier-1 (LERP fallback): near-parallel tensors interpolate linearly (no NaN from
// sin Ω → 0).
func TestSLERPLerpFallback(t *testing.T) {
	a := toT([][]float64{{1, 2, 3}})
	b := toT([][]float64{{1.00001, 2.00002, 3.00003}}) // essentially parallel ⇒ LERP
	got, _ := nn.SLERP(a, b, 0.5)
	for i := range got.Numel() {
		c := tensor.Unravel(i, got.Shape())
		want := 0.5*a.AtF64(c...) + 0.5*b.AtF64(c...)
		if math.IsNaN(got.AtF64(c...)) || math.Abs(got.AtF64(c...)-want) > 1e-9 {
			t.Errorf("parallel fallback at %v: %g want LERP %g", c, got.AtF64(c...), want)
		}
	}
}

func TestSLERPErrors(t *testing.T) {
	if _, err := nn.SLERP(tensor.New(tensor.F64, tensor.Shape{2, 3}), tensor.New(tensor.F64, tensor.Shape{3, 2}), 0.5); err == nil {
		t.Error("shape mismatch: want error")
	}
}

func ExampleSLERP() {
	// Halfway between two orthogonal unit vectors: SLERP stays on the unit sphere (norm 1),
	// where plain averaging would give norm ≈ 0.71.
	a := toT([][]float64{{1, 0}})
	b := toT([][]float64{{0, 1}})
	mid, _ := nn.SLERP(a, b, 0.5)
	fmt.Printf("%.3f\n", norm(mid))
	// Output:
	// 1.000
}
