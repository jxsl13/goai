package linalg_test

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// svdDigest hashes every bit of U, s and V. A tolerance comparison is the wrong gate for a
// transform that claims to change no value: Jacobi is iterative, so a reassociated reduction
// still converges and still passes any reasonable tolerance while returning different bits.
// Only a bit-exact digest can tell the two apart.
func svdDigest(t *testing.T, m, n int) uint64 {
	t.Helper()
	d := make([]float64, m*n)
	for k := range d {
		// Deterministic, ill-conditioned enough that the sweeps actually rotate, and with
		// near-duplicate columns so some pairs converge early and skip their rotation —
		// the skip path is what a cached norm has to stay correct across.
		i, j := k/n, k%n
		d[k] = math.Sin(float64(i*31+j*7))*float64(1+j%3) + float64(i%5)*0.25
	}
	u, s, v, err := linalg.SVD(tensor.FromFloat64(tensor.Shape{m, n}, d))
	if err != nil {
		t.Fatalf("%dx%d: %v", m, n, err)
	}
	h := uint64(14695981039346656037)
	for _, tn := range []*tensor.Tensor{u, s, v} {
		for _, x := range tn.Storage().F64() {
			b := math.Float64bits(x)
			for sh := 0; sh < 64; sh += 8 {
				h = (h ^ (b>>sh)&0xff) * 1099511628211
			}
		}
	}
	return h
}

// TestSVDIsBitIdentical freezes the factorization of five shapes, including the m < n path
// that recurses through the transpose and a single-column case with no pair to rotate at all.
//
// The digests were taken from the three-accumulator form that recomputed both column norms
// inside every pair. They must survive maintaining those norms incrementally instead: the
// update runs over ascending k with one accumulator on exactly the rotated values, which is
// operand-for-operand what the recomputation did, so nothing may move even in the last bit.
func TestSVDIsBitIdentical(t *testing.T) {
	if raceEnabled {
		// The digest is build-mode dependent here and only here; see race_on_test.go.
		t.Skip("SVD digest is not stable under -race on arm64 (FMA contraction differs)")
	}
	cases := []struct {
		m, n int
		want uint64
	}{
		{8, 8, archgold.Pick(3416335863526090039, 7028714950098526873)},
		{32, 12, archgold.Pick(16807871276312544421, 11094837223896545758)},
		{12, 32, archgold.Pick(12459786375205486409, 3100706285955080142)}, // m < n: recurses on the transpose
		{64, 64, archgold.Pick(5080245599646072399, 14914027006611402415)},
		{40, 1, archgold.Pick(18129668683049372422, 18129668683049372422)}, // no pair exists; the sweep body never runs
	}
	for _, c := range cases {
		got := svdDigest(t, c.m, c.n)
		if got != c.want {
			t.Fatalf("%dx%d digest %d, want %d", c.m, c.n, got, c.want)
		}
	}
}
