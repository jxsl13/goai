package backend

import (
	"fmt"
	"math"
	"testing"
)

// §R60: for a power-of-two head count the ALiBi slopes are the geometric sequence
// 2^(−8·(h+1)/n) — for n=8 that is 2^−1 … 2^−8.
func TestALiBiSlopesPowerOfTwo(t *testing.T) {
	got := ALiBiSlopes(8)
	for h := range 8 {
		want := math.Exp2(-float64(h + 1)) // 2^(-8(h+1)/8) = 2^-(h+1)
		if math.Abs(got[h]-want) > 1e-15 {
			t.Errorf("slope[%d] = %.12g, want %.12g", h, got[h], want)
		}
	}
	// ratio between consecutive slopes is constant 2^(-8/n) = 1/2 for n=8
	for h := 1; h < 8; h++ {
		if r := got[h] / got[h-1]; math.Abs(r-0.5) > 1e-15 {
			t.Errorf("slope ratio at %d = %.6f, want 0.5", h, r)
		}
	}
}

// §R60 non-power-of-two: n=3 takes the two slopes of the closest power of two (2)
// then one interleaved slope from the next power of two (4).
func TestALiBiSlopesNonPowerOfTwo(t *testing.T) {
	got := ALiBiSlopes(3)
	// closest=2 → [2^-4, 2^-8]; extra from n=4 → [2^-2,2^-4,2^-6,2^-8], take extra[0]=2^-2
	want := []float64{math.Exp2(-4), math.Exp2(-8), math.Exp2(-2)}
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-15 {
			t.Errorf("slope[%d] = %.12g, want %.12g", i, got[i], want[i])
		}
	}
}

// Slopes are strictly positive and decreasing (each head penalizes distance less
// aggressively than the previous within a power-of-two block).
func TestALiBiSlopesPositive(t *testing.T) {
	for _, n := range []int{1, 2, 4, 5, 8, 16} {
		for _, s := range ALiBiSlopes(n) {
			if s <= 0 {
				t.Errorf("n=%d: non-positive slope %g", n, s)
			}
		}
	}
}

func ExampleALiBiSlopes() {
	// The per-head ALiBi slopes for a 4-head model: a geometric sequence with
	// ratio 2^(-8/4)=1/4, i.e. 2^-2, 2^-4, 2^-6, 2^-8. Each head applies a
	// different linear recency bias to the attention scores.
	for _, s := range ALiBiSlopes(4) {
		fmt.Printf("%.4f ", s)
	}
	fmt.Println()
	// Output: 0.2500 0.0625 0.0156 0.0039
}
