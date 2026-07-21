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

func TestSoftplusNegLLSumF64Parity(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 16, 17, 1000} {
		f := make([]float64, n)
		y := make([]float64, n)
		for i := range f {
			f[i] = -20 + 40*float64(i)/float64(max(n-1, 1)) // ±20 (both saturation tails)
			if i%2 == 0 {
				y[i] = 1
			}
		}
		got := SoftplusNegLLSumF64(f, y)
		var want float64
		for i := range f {
			x := (1 - 2*y[i]) * f[i]
			if x > 0 {
				want += x + math.Log1p(math.Exp(-x))
			} else {
				want += math.Log1p(math.Exp(x))
			}
		}
		if rel := math.Abs(got-want) / math.Max(1e-300, math.Abs(want)); rel > 1e-13 {
			t.Fatalf("n=%d: got=%.15g want=%.15g rel=%.2e", n, got, want, rel)
		}
	}
}
