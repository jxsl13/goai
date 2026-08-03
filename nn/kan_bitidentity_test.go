package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestKANForwardIsBitIdentical freezes the layer's output. Banding the fused spline over the
// batch claims to change no value — a band never splits a row, so each row's accumulation over
// the input dimension keeps the same ascending order — and a KAN is a smooth approximator whose
// outputs would absorb a reassociation without any test noticing.
//
// The batch sizes straddle the fan-out gate: the 3-row case runs serially in both arms, the
// 96-row case bands, and 13 is deliberately not a multiple of the worker count so the last
// band is short.
func TestKANForwardIsBitIdentical(t *testing.T) {
	cases := []struct {
		batch, in, out int
		want           uint64
	}{
		{3, 5, 7, 5936029728971432568},
		{13, 8, 6, 15159748691548848689},
		{96, 24, 32, 515177776064738749},
	}
	for _, c := range cases {
		l, err := NewKAN(c.in, c.out, 1)
		if err != nil {
			t.Fatal(err)
		}
		x := tensor.New(tensor.F64, tensor.Shape{c.batch, c.in})
		xs := x.Storage().F64()
		for i := range xs {
			xs[i] = math.Sin(float64(i*11+5)) * 0.6
		}
		y, err := l.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		for _, v := range y.Storage().F64() {
			b := math.Float64bits(v)
			for s := 0; s < 64; s += 8 {
				h = (h ^ (b>>s)&0xff) * 1099511628211
			}
		}
		if h != c.want {
			t.Fatalf("%dx%dx%d digest %d, want %d", c.batch, c.in, c.out, h, c.want)
		}
	}
}
