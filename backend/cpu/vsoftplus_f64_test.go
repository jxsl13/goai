//go:build amd64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"
)

// refSoftplus is the numerically-stable scalar reference (identical to ref's
// softplusKernel / nlp mamba softplus): max(x,0) + log1p(e^(−|x|)).
func refSoftplus(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}

// TestVsoftplusF64Accuracy checks vsoftplusF64 vs the stable scalar reference over a
// wide range. softplus feeds the Mamba/Jamba Δ; the goldens gate at 1e-9, so a
// ~1e-13 kernel is comfortably inside.
func TestVsoftplusF64Accuracy(t *testing.T) {
	n := 4096
	src := make([]float64, n)
	for i := range src {
		src[i] = -60 + 120*float64(i)/float64(n) // [-60, 60]
	}
	dst := make([]float64, n)
	vsoftplusF64(dst, src)
	var maxRel, maxAbs float64
	for i, x := range src {
		want := refSoftplus(x)
		abs := math.Abs(dst[i] - want)
		den := math.Max(1, math.Abs(want))
		rel := abs / den
		if rel > maxRel {
			maxRel = rel
		}
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	t.Logf("vsoftplusF64 vs scalar over [-60,60] n=%d: maxRel=%.3e maxAbs=%.3e", n, maxRel, maxAbs)
	if maxRel > 1e-13 {
		t.Fatalf("vsoftplusF64 maxRel=%.3e exceeds 1e-13", maxRel)
	}
}

// TestSoftplusF64polyMatchesVector asserts the scalar tail is BIT-EXACT to the SIMD
// lane, so a value yields the same softplus whether it lands in the 4-lane body or
// the len%4 tail (mirrors TestExpF64polyMatchesVector).
func TestSoftplusF64polyMatchesVector(t *testing.T) {
	mism := 0
	for x := -60.0; x <= 60.0; x += 0.00017 {
		// body: process a 4-lane block, take lane 0.
		body := make([]float64, 4)
		vsoftplusF64(body, []float64{x, x, x, x})
		s := softplusF64poly(x)
		if math.Float64bits(body[0]) != math.Float64bits(s) {
			if mism < 5 {
				t.Errorf("x=%v: vector %v (%#x) != scalar %v (%#x)", x, body[0], math.Float64bits(body[0]), s, math.Float64bits(s))
			}
			mism++
		}
	}
	if mism > 0 {
		t.Fatalf("%d bit-mismatches between softplusF64x4 body and softplusF64poly", mism)
	}
}

// TestVsoftcapF64Accuracy checks vsoftcapF64 vs the scalar reference cap·tanh(x/cap)
// (Gemma-2 soft-cap). The goldens gate at model f64 tolerance, so ~1e-13 is inside.
func TestVsoftcapF64Accuracy(t *testing.T) {
	n := 4096
	for _, cap := range []float64{30.0, 50.0} { // Gemma-2 attn=50, final=30
		src := make([]float64, n)
		for i := range src {
			src[i] = -400 + 800*float64(i)/float64(n) // spans well past ±cap (saturation)
		}
		dst := make([]float64, n)
		vsoftcapF64(dst, src, cap)
		var maxRel float64
		for i, x := range src {
			want := cap * math.Tanh(x/cap)
			den := math.Max(1, math.Abs(want))
			if rel := math.Abs(dst[i]-want) / den; rel > maxRel {
				maxRel = rel
			}
		}
		t.Logf("vsoftcapF64 cap=%.0f vs scalar over [-400,400] n=%d: maxRel=%.3e", cap, n, maxRel)
		if maxRel > 1e-13 {
			t.Fatalf("vsoftcapF64 cap=%.0f maxRel=%.3e exceeds 1e-13", cap, maxRel)
		}
	}
}

// TestSoftcapF64polyMatchesVector asserts the scalar tail is BIT-EXACT to the SIMD
// lane (batch-length-independent soft-cap, mirrors the softplus/exp twins).
func TestSoftcapF64polyMatchesVector(t *testing.T) {
	const cap = 50.0
	mism := 0
	for x := -400.0; x <= 400.0; x += 0.0013 {
		body := make([]float64, 4)
		vsoftcapF64(body, []float64{x, x, x, x}, cap)
		s := softcapF64poly(x, cap)
		if math.Float64bits(body[0]) != math.Float64bits(s) {
			if mism < 5 {
				t.Errorf("x=%v: vector %v (%#x) != scalar %v (%#x)", x, body[0], math.Float64bits(body[0]), s, math.Float64bits(s))
			}
			mism++
		}
	}
	if mism > 0 {
		t.Fatalf("%d bit-mismatches between vsoftcapF64 body and softcapF64poly", mism)
	}
}
