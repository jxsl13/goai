//go:build goexperiment.simd

package cpu

import (
	"math"
	"testing"
)

// erfF64poly must match math.Erf within ~1 ulp across the whole range, including the
// branch boundaries |y|=1 and |y|=6 and the negative half.
func TestErfF64PolyAccuracy(t *testing.T) {
	var maxRel float64
	for x := -8.0; x <= 8.0; x += 1.0 / 512 {
		got := erfF64poly(x)
		want := math.Erf(x)
		d := math.Abs(got - want)
		if want != 0 {
			d /= math.Abs(want)
		}
		if d > maxRel {
			maxRel = d
		}
	}
	t.Logf("erfF64poly max rel err = %.3e", maxRel)
	if maxRel > 1e-13 {
		t.Fatalf("erfF64poly too inaccurate: max rel err %.3e > 1e-13", maxRel)
	}
}

// vgeluF64 (SIMD body + scalar tail) must match the reference GELU 0.5·x·(1+erf(x/√2))
// within the model f64 tolerance, and the body must equal the tail (length-independent).
func TestVGeluF64Accuracy(t *testing.T) {
	ref := func(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }
	// 4099 = 1024 full 4-lanes + a 3-element scalar tail, so both paths are exercised.
	n := 4099
	src := make([]float64, n)
	for i := range src {
		src[i] = -12 + 24*float64(i)/float64(n-1) // span [-12,12], crossing |y|=1 (x≈±1.41) and |y|=6 (x≈±8.49)
	}
	dst := make([]float64, n)
	vgeluF64(dst, src)
	var maxAbs float64
	for i, x := range src {
		w := ref(x)
		tol := 1e-12 * math.Max(1, math.Abs(w))
		if d := math.Abs(dst[i] - w); d > maxAbs {
			maxAbs = d
		}
		if math.Abs(dst[i]-w) > tol {
			t.Fatalf("vgeluF64 x=%g: got %.17g want %.17g (Δ=%.3e > tol %.3e)", x, dst[i], w, math.Abs(dst[i]-w), tol)
		}
	}
	t.Logf("vgeluF64 max abs err vs ref = %.3e", maxAbs)

	// body == tail: GELU of the same value must be identical whether it lands in a
	// 4-lane body or the scalar remainder (byte-exact, any length/alignment).
	for _, x := range []float64{-3.3, -1.4142135623730951, -0.5, 0, 0.7, 1.4142135623730951, 3.0, 8.485281374238571, 9.5} {
		one := make([]float64, 1)
		vgeluF64(one, []float64{x})
		four := make([]float64, 4)
		vgeluF64(four, []float64{x, x, x, x})
		if one[0] != four[0] {
			t.Errorf("body!=tail at x=%g: tail %.17g vs body %.17g", x, one[0], four[0])
		}
	}
}

// vgeluGradF64 must match the exact reference dx = g·(Φ(x)+x·φ(x)) within the model f64
// tolerance, and body must equal tail.
func TestVGeluGradF64Accuracy(t *testing.T) {
	ref := func(x, g float64) float64 {
		phi := 0.5 * (1 + math.Erf(x/math.Sqrt2))
		pdf := 0.3989422804014327 * math.Exp(-0.5*x*x)
		return g * (phi + x*pdf)
	}
	n := 4099
	xs := make([]float64, n)
	gs := make([]float64, n)
	for i := range xs {
		xs[i] = -12 + 24*float64(i)/float64(n-1)
		gs[i] = -2 + 4*float64((i*37)%n)/float64(n-1)
	}
	dst := make([]float64, n)
	vgeluGradF64(dst, xs, gs)
	var maxAbs float64
	for i := range xs {
		w := ref(xs[i], gs[i])
		if d := math.Abs(dst[i] - w); d > maxAbs {
			maxAbs = d
		}
		if math.Abs(dst[i]-w) > 1e-12*math.Max(1, math.Abs(w)) {
			t.Fatalf("vgeluGradF64 x=%g g=%g: got %.17g want %.17g", xs[i], gs[i], dst[i], w)
		}
	}
	t.Logf("vgeluGradF64 max abs err vs ref = %.3e", maxAbs)
	for _, x := range []float64{-3.3, -1.4142135623730951, 0, 1.4142135623730951, 8.49, 9.5} {
		one := make([]float64, 1)
		vgeluGradF64(one, []float64{x}, []float64{1.7})
		four := make([]float64, 4)
		vgeluGradF64(four, []float64{x, x, x, x}, []float64{1.7, 1.7, 1.7, 1.7})
		if one[0] != four[0] {
			t.Errorf("grad body!=tail at x=%g: tail %.17g vs body %.17g", x, one[0], four[0])
		}
	}
}
