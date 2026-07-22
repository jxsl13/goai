package nn

import (
	"math"
	"testing"
)

// §V16 tier-1 property: the factored second moment V̂ = R⊗C/ΣR is EXACT when g² is
// rank-1 (Shazeer & Stern §3 / Lemma 1) — the assumption the memory saving rests
// on. With g²[i,j] = a[i]·b[j] the row/column sums R,C reconstruct g² bit-close.
// (Internal test: exercises the unexported reconstruction directly.)
func TestAdafactorFactoredExactRank1(t *testing.T) {
	a := []float64{0.5, 2.0, 1.5, 3.0}
	b := []float64{1.0, 0.25, 4.0}
	var sumA, sumB float64
	for _, v := range a {
		sumA += v
	}
	for _, v := range b {
		sumB += v
	}
	// R[i] = Σ_j a[i]b[j] = a[i]·Σb ; C[j] = Σ_i a[i]b[j] = b[j]·Σa
	r := make([]float64, len(a))
	c := make([]float64, len(b))
	for i := range a {
		r[i] = a[i] * sumB
	}
	for j := range b {
		c[j] = b[j] * sumA
	}
	vhat := make([]float64, len(r)*len(c))
	adafactorVhat(vhat, r, c)
	for i := range a {
		for j := range b {
			want := a[i] * b[j] // = g²[i,j]
			if got := vhat[i*len(b)+j]; math.Abs(got-want) > 1e-12*math.Max(1, want) {
				t.Errorf("V̂[%d,%d] = %v, want rank-1 g² = %v", i, j, got, want)
			}
		}
	}
}
