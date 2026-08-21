package simd

import (
	"math"
	"slices"
	"testing"
)

func sameF64Within(got, want, relTol float64) bool {
	if math.IsNaN(want) {
		return math.IsNaN(got)
	}
	if got == want {
		return true
	}
	return math.Abs(got-want) <= relTol*math.Max(1e-300, math.Abs(want))
}

func TestExpScaledF64Parity(t *testing.T) {
	// A<0, Δ≥0 → the SSM scan argument scale·src is ≤ 0. Sweep lengths incl. non-
	// multiples of 4 (the len%4 tail) and both cheap/saturating magnitudes.
	for _, n := range []int{1, 3, 4, 7, 16, 17, 100, 4096} {
		for _, scale := range []float64{0.05, 1.0, 8.0} {
			src := make([]float64, n)
			for i := range src {
				src[i] = -3 * float64(i) / float64(max(n-1, 1)) // A ∈ [-3, 0]
			}
			dst := make([]float64, n)
			original := slices.Clone(src)
			ExpScaledF64(dst, src, scale)
			if !slices.Equal(src, original) {
				t.Fatalf("n=%d scale=%g: source was modified", n, scale)
			}
			var maxRel float64
			for i, v := range src {
				w := math.Exp(scale * v)
				if rel := math.Abs(dst[i]-w) / math.Max(1e-300, w); rel > maxRel {
					maxRel = rel
				}
			}
			if maxRel > 1e-13 {
				t.Fatalf("n=%d scale=%g: maxRel=%.2e exceeds 1e-13", n, scale, maxRel)
			}
		}
	}
}

func TestExpSumF64Parity(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 100, 4096, 128000} {
		src := make([]float64, n)
		for i := range src {
			src[i] = -40 + 40*float64(i)/float64(max(n-1, 1)) // [-40, 0], softmax range
		}
		bias := 0.0
		for _, v := range src {
			if v > bias {
				bias = v
			}
		}
		dst := make([]float64, n)
		sum := ExpSumF64(dst, src, bias)
		var wantSum, maxRel float64
		for i, v := range src {
			w := math.Exp(v - bias)
			wantSum += w
			den := math.Max(1e-300, w)
			if rel := math.Abs(dst[i]-w) / den; rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > 1e-13 {
			t.Fatalf("n=%d: dst maxRel=%.2e exceeds 1e-13", n, maxRel)
		}
		if rel := math.Abs(sum-wantSum) / wantSum; rel > 1e-13 {
			t.Fatalf("n=%d: sum rel=%.2e exceeds 1e-13", n, rel)
		}
	}
}

func TestExpSumF64InPlaceOddAndInputImmutability(t *testing.T) {
	src := []float64{-7, -4, -2, -1, -0.5, -0.25, 0}
	wantSrc := slices.Clone(src)
	dst := make([]float64, len(src))
	gotSum := ExpSumF64(dst, src, 0)
	if !slices.Equal(src, wantSrc) {
		t.Fatal("distinct source was modified")
	}
	var wantSum float64
	for i, x := range wantSrc {
		want := math.Exp(x)
		wantSum += want
		if !sameF64Within(dst[i], want, 1e-13) {
			t.Fatalf("distinct dst[%d]=%g want %g", i, dst[i], want)
		}
	}
	if !sameF64Within(gotSum, wantSum, 1e-13) {
		t.Fatalf("distinct sum=%g want %g", gotSum, wantSum)
	}

	inPlace := slices.Clone(wantSrc)
	gotSum = ExpSumF64(inPlace, inPlace, 0)
	for i, x := range wantSrc {
		want := math.Exp(x)
		if !sameF64Within(inPlace[i], want, 1e-13) {
			t.Fatalf("in-place dst[%d]=%g want %g", i, inPlace[i], want)
		}
	}
	if !sameF64Within(gotSum, wantSum, 1e-13) {
		t.Fatalf("in-place sum=%g want %g", gotSum, wantSum)
	}
}
