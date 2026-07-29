package simd

import (
	"math"
	"math/rand/v2"
	"testing"
)

// fwhtRef is the canonical scalar butterfly oracle.
func fwhtRef(a []float64) {
	n := len(a)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				x, y := a[j], a[j+h]
				a[j], a[j+h] = x+y, x-y
			}
		}
	}
}

// TestFWHTF64BitExact locks FWHTF64 (vectorized on simd builds) to the scalar oracle,
// bit-for-bit, across power-of-two sizes incl. the h<4 scalar-only and h>=4 vectorized ranges.
func TestFWHTF64BitExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{1, 2, 4, 8, 16, 128, 512, 4096} {
		a := make([]float64, n)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		want := make([]float64, n)
		copy(want, a)
		fwhtRef(want)
		got := make([]float64, n)
		copy(got, a)
		FWHTF64(got)
		for i := range got {
			if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
				t.Fatalf("n=%d idx=%d got %v want %v (not bit-identical)", n, i, got[i], want[i])
			}
		}
	}
}
