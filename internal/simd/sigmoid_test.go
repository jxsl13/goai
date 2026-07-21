package simd

import (
	"math"
	"testing"
)

func TestSigmoidF64Parity(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 16, 17, 1000} {
		src := make([]float64, n)
		for i := range src {
			src[i] = -30 + 60*float64(i)/float64(max(n-1, 1)) // spans ±30 (saturation both ends)
		}
		dst := make([]float64, n)
		SigmoidF64(dst, src)
		var maxRel float64
		for i, x := range src {
			w := 1 / (1 + math.Exp(-x))
			if rel := math.Abs(dst[i]-w) / math.Max(1e-300, w); rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > 1e-13 {
			t.Fatalf("n=%d: maxRel=%.2e exceeds 1e-13", n, maxRel)
		}
	}
}
