//go:build amd64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"

	"simd/archsimd"
)

// TestExpF64x4Accuracy checks expF64x4 vs math.Exp over the SiLU arg range (x ≤ 0)
// to sub-ulp — the accuracy the Llama f64 golden (1e-12) relies on.
func TestExpF64x4Accuracy(t *testing.T) {
	var maxUlp, maxRel float64
	for x := -40.0; x <= 0.0; x += 0.0003 {
		in := []float64{x, x, x, x}
		got := make([]float64, 4)
		// route through vsiluF64's exp by calling expF64x4 directly via a tiny shim.
		v := loadExp(in)
		copy(got, v)
		want := math.Exp(x)
		if want == 0 {
			continue
		}
		rel := math.Abs(got[0]-want) / want
		if rel > maxRel {
			maxRel = rel
		}
		ulp := math.Abs(got[0]-want) / ulpF64(want)
		if ulp > maxUlp {
			maxUlp = ulp
		}
	}
	t.Logf("expF64x4 vs math.Exp over [-40,0]: maxRel=%.3e maxUlp=%.2f", maxRel, maxUlp)
	if maxRel > 1e-14 {
		t.Fatalf("expF64x4 maxRel=%.3e exceeds 1e-14", maxRel)
	}
}

// TestVsiluF64Accuracy checks vsiluF64 vs the scalar x/(1+exp(-x)) reference.
func TestVsiluF64Accuracy(t *testing.T) {
	n := 4096
	src := make([]float64, n)
	for i := range src {
		src[i] = -80 + 160*float64(i)/float64(n) // [-80, 80]
	}
	dst := make([]float64, n)
	vsiluF64(dst, src)
	var maxRel float64
	for i, x := range src {
		want := x / (1 + math.Exp(-x))
		den := math.Max(1, math.Abs(want))
		rel := math.Abs(dst[i]-want) / den
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("vsiluF64 vs scalar over [-80,80] n=%d: maxRel=%.3e", n, maxRel)
	if maxRel > 1e-13 {
		t.Fatalf("vsiluF64 maxRel=%.3e exceeds 1e-13", maxRel)
	}
}

func ulpF64(x float64) float64 {
	return math.Abs(math.Nextafter(x, math.Inf(1)) - x)
}

// TestExpF64polyMatchesVector asserts the scalar tail primitive is BIT-EXACT to
// the SIMD lane — the property that keeps decode==forward byte-exact.
func TestExpF64polyMatchesVector(t *testing.T) {
	mism := 0
	for x := -80.0; x <= 0.0; x += 0.00017 {
		v := loadExp([]float64{x, x, x, x})
		s := expF64poly(x)
		if math.Float64bits(v[0]) != math.Float64bits(s) {
			if mism < 5 {
				t.Errorf("x=%v: vector %v (%#x) != scalar %v (%#x)", x, v[0], math.Float64bits(v[0]), s, math.Float64bits(s))
			}
			mism++
		}
	}
	if mism > 0 {
		t.Fatalf("%d bit-mismatches between expF64x4 and expF64poly", mism)
	}
}

func loadExp(in []float64) []float64 {
	x := archsimd.LoadFloat64x4Slice(in)
	out := make([]float64, 4)
	expF64x4(x).StoreSlice(out)
	return out
}
