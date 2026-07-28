package cpu

import (
	"math"
	"testing"
)

// TestVsigmoidF64Accuracy checks the 4-wide σ kernel against scalar 1/(1+e^-x) over a wide
// range, at the same 1e-13 relative bar the sibling vsiluF64/vsoftcapF64 F64 SIMD wins ride.
func TestVsigmoidF64Accuracy(t *testing.T) {
	n := 4096
	src := make([]float64, n)
	for i := range src {
		src[i] = -80 + 160*float64(i)/float64(n) // [-80, 80]
	}
	dst := make([]float64, n)
	vsigmoidF64(dst, src)
	var maxRel float64
	for i, x := range src {
		want := 1 / (1 + math.Exp(-x))
		den := math.Max(1, math.Abs(want))
		rel := math.Abs(dst[i]-want) / den
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("vsigmoidF64 vs scalar over [-80,80] n=%d: maxRel=%.3e", n, maxRel)
	if maxRel > 1e-13 {
		t.Fatalf("vsigmoidF64 maxRel=%.3e exceeds 1e-13", maxRel)
	}
}
