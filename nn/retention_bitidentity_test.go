package nn_test

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// RetentionRecurrent carries a state across steps, so any reordering compounds: an error at
// step 0 rides through every later output rather than staying local. Both of the loops jammed
// here run over the KEY dimension — one updating the state, one summing the output — and both
// claim the same key order as before. A recurrence is exactly where a tolerance test stops
// meaning anything, so this hashes bits.
func retentionRecurrentDigest(t *testing.T, l, dk, dv int, gamma float64) uint64 {
	t.Helper()
	mk := func(c int, seed int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{l, c})
		s := x.Storage().F64()
		for i := range s {
			s[i] = math.Sin(float64(i*7+seed*11)) * 0.4
		}
		return x
	}
	out, err := nn.RetentionRecurrent(mk(dk, 1), mk(dk, 2), mk(dv, 3), gamma)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	n := out.Numel()
	for _, val := range out.Storage().F64()[:n] {
		u := math.Float64bits(val)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	return h
}

func TestRetentionRecurrentIsBitIdentical(t *testing.T) {
	// dk is the jammed dimension, so it carries the remainders: 13 is odd and prime, 22 leaves
	// 6 modulo 8, 32 divides 2, 4 and 8 so the tail never runs, and dk=1 skips the jam
	// entirely. Several steps in each case so the state recurrence is exercised, not just the
	// first update.
	cases := []struct {
		l, dk, dv int
		gamma     float64
		want      uint64
	}{
		{7, 13, 5, 0.968, archgold.Pick(11973786104883440334, 788445753469414663)},
		{5, 22, 9, 0.9, archgold.Pick(6912945384076962978, 14580955136303493942)},
		{4, 32, 8, 0.968, archgold.Pick(11555021637882473957, 14113188775936440872)},
		{6, 1, 4, 0.5, archgold.Pick(802478018091383884, 16131837705487315143)},
		{3, 8, 1, 1.0, archgold.Pick(14149284330353845558, 10768318184156327447)}, // gamma=1: no decay, so the state is a running sum
	}
	for _, c := range cases {
		got := retentionRecurrentDigest(t, c.l, c.dk, c.dv, c.gamma)
		if got != c.want {
			t.Errorf("l=%d dk=%d dv=%d gamma=%v: digest %d", c.l, c.dk, c.dv, c.gamma, got)
		}
	}
}
