package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// SSDRecurrent carries an [N,d] state across steps, so a one-ulp drift compounds rather than
// staying local — the same reason RetentionRecurrent is gated on bits and not on a tolerance.
//
// The shapes below straddle the kernel's OWN layout switch as well as the jam's remainders:
// n*d >= 4096 takes the interleaved output path and anything smaller takes the strided one,
// and only the interleaved path contains the jammed loop. A test that stayed under the
// threshold would exercise none of this.
func ssdRecurrentDigest(t *testing.T, T, d, n int) uint64 {
	t.Helper()
	mk := func(rows, cols, seed int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		s := x.Storage().F64()
		for i := range s {
			s[i] = math.Sin(float64(i*7+seed*11)) * 0.4
		}
		return x
	}
	a := tensor.New(tensor.F64, tensor.Shape{T})
	as := a.Storage().F64()
	for i := range as {
		as[i] = 0.9 + 0.05*math.Cos(float64(i)*0.02)
	}
	out, err := nn.SSDRecurrent(mk(T, d, 1), a, mk(T, n, 2), mk(T, n, 3))
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	numel := out.Numel()
	for _, v := range out.Storage().F64()[:numel] {
		u := math.Float64bits(v)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	return h
}

func TestSSDRecurrentIsBitIdentical(t *testing.T) {
	// n is the jammed dimension: 13 is odd and prime, 70 leaves 6 modulo 8, 64 divides every
	// width so the tail never runs, and n=1 skips the jam entirely. The first four cross the
	// 4096 threshold into the interleaved path; the last stays under it as a control on the
	// path this change does not touch.
	cases := []struct {
		T, d, n int
		want    uint64
	}{
		{5, 320, 13, 14628096508942058614},
		{4, 64, 70, 16279910178519077628},
		{4, 64, 64, 16083548054130141405},
		{6, 4096, 1, 11038940808816861172},
		{5, 16, 8, 4559563584542277843}, // n*d = 128: strided path, untouched by this round
	}
	for _, c := range cases {
		if got := ssdRecurrentDigest(t, c.T, c.d, c.n); got != c.want {
			t.Errorf("T=%d d=%d n=%d (n*d=%d): digest %d", c.T, c.d, c.n, c.d*c.n, got)
		}
	}
}
