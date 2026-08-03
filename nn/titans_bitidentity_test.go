package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestTitansScanIsBitIdentical freezes every bit of the deep scan's output. Jamming its four
// inner loops four units per pass claims to change no value — each unit keeps its own
// accumulator over the same ascending index — and the companion dispatch test is a TOLERANCE
// relationship (the two paths already differ by an ulp on this host), so nothing else in the
// suite would notice a reassociation.
//
// The shapes straddle both jams: dim and hid divisible by four, then remainders of two and one,
// then three and three.
func TestTitansScanIsBitIdentical(t *testing.T) {
	for _, c := range []struct {
		seq, dim, hid int
		want          uint64
	}{
		{16, 8, 12, 8353175572084491654},
		{7, 10, 13, 16808472780882284121},
		{9, 14, 7, 5379503336826529122},
	} {
		m, err := nn.NewNeuralMemory(tensor.F64, c.dim, c.hid, 3)
		if err != nil {
			t.Fatal(err)
		}
		mk := func(off int) *tensor.Tensor {
			tn := tensor.New(tensor.F64, tensor.Shape{c.seq, c.dim})
			s := tn.Storage().F64()
			for i := range s {
				s[i] = math.Sin(float64(i*7+off)) * 0.75
			}
			return tn
		}
		sc := func(off int) *tensor.Tensor {
			tn := tensor.New(tensor.F64, tensor.Shape{c.seq, 1})
			s := tn.Storage().F64()
			for i := range s {
				s[i] = 0.25 + 0.5*math.Abs(math.Cos(float64(i*3+off)))
			}
			return tn
		}
		out, err := m.Scan(backend.NewContext(), mk(1), mk(2), mk(3), sc(4), sc(5), sc(6))
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		os := out.Contiguous().Storage().F64()
		for _, v := range os {
			u := math.Float64bits(v)
			for s := 0; s < 64; s += 8 {
				h = (h ^ (u>>s)&0xff) * 1099511628211
			}
		}
		if h != c.want {
			t.Fatalf("seq=%d dim=%d hid=%d digest %d, want %d", c.seq, c.dim, c.hid, h, c.want)
		}
	}
}
