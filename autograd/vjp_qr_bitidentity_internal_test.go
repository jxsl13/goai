package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The QR backward's dominant term is a rank-1 update accumulated over the m rows of Q:
// mm[i][j] -= Qbar[k][i]*Q[k][j], with mm shared across k. Jamming k so mm[i][j] is held in a
// register across several subtractions claims to change nothing — the same terms in the same
// ascending k — and a gradient check would absorb a small change without complaint, so the
// digest is the gate.
//
// The subtraction order is the whole claim, which is why this hashes BITS. Floating-point
// subtraction is not associative, and the jam is only correct if each mm[i][j] still sees k
// ascending; a version that summed per-k partials and folded them afterward would pass any
// tolerance test and fail here.
func qrVJPDigest(t *testing.T, m, n int) uint64 {
	t.Helper()
	vjp := vjpsMulti[backend.OpQR]
	if vjp == nil {
		t.Fatal("no multi-output VJP registered for OpQR")
	}
	mk := func(rows, cols int, f func(i, j int) float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		for i := range rows {
			for j := range cols {
				x.SetF64(f(i, j), i, j)
			}
		}
		return x
	}
	q := mk(m, n, func(i, j int) float64 {
		return math.Sqrt(2/float64(m)) * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(m))
	})
	r := mk(n, n, func(i, j int) float64 {
		switch {
		case i > j:
			return 0
		case i == j:
			return 2 + 0.5*float64(i%7)
		}
		return math.Sin(float64(i*3+j*5)) * 0.4
	})
	qbar := mk(m, n, func(i, j int) float64 { return math.Cos(float64(i*7+j*3)) * 0.3 })
	rbar := mk(n, n, func(i, j int) float64 {
		if i > j {
			return 0
		}
		return math.Sin(float64(i*5+j*11)) * 0.25
	})
	gs, err := vjp(nil, nil, []*tensor.Tensor{q, r}, nil, []*tensor.Tensor{qbar, rbar})
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	for _, g := range gs {
		nEl := g.Numel()
		for _, v := range g.Storage().F64()[:nEl] {
			u := math.Float64bits(v)
			for s := 0; s < 64; s += 8 {
				h = (h ^ (u>>s)&0xff) * 1099511628211
			}
		}
	}
	return h
}

func TestQRVJPIsBitIdentical(t *testing.T) {
	// The row counts straddle every remainder a jam of 2, 4 or 8 can leave: 13 is odd and
	// prime, 22 leaves 6 modulo 8, 32 divides all three so the tail is skipped, and m=3 with
	// n=3 is small enough that the jammed loop never runs at all.
	cases := []struct {
		m, n int
		want uint64
	}{
		{13, 5, 1026364614881149619},
		{22, 7, 12476853988711379297},
		{32, 8, 14029057225193316077},
		{3, 3, 6149150020269390530},
		{64, 32, 15802176959280348119},
	}
	for _, c := range cases {
		if got := qrVJPDigest(t, c.m, c.n); got != c.want {
			t.Errorf("m=%d n=%d: digest %d", c.m, c.n, got)
		}
	}
}
