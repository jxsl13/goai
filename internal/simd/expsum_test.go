package simd

import (
	"math"
	"testing"
)

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
