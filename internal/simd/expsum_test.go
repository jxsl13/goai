package simd

import (
	"math"
	"testing"
)

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
			ExpScaledF64(dst, src, scale)
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
