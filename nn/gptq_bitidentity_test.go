package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestGPTQuantizeIsBitIdentical freezes the quantized weight. Interchanging the sweep to
// row-outer and banding it claims to change no value — rows never observe each other, and
// within a row the columns are still ascending — and a quantizer is somewhere a changed
// rounding decision would look like a different-but-plausible grid snap.
//
// One shape is below the fan-out gate and two clear it. Two quantizers are used, because the
// callback is now invoked from several goroutines and a coarser grid changes which columns
// carry error into the compensation.
func TestGPTQuantizeIsBitIdentical(t *testing.T) {
	cases := []struct {
		out, in, samples, levels int
		want                     uint64
	}{
		{4, 8, 16, 16, 9062784375536821882},
		{48, 96, 64, 16, 6027463531619723289},
		{48, 96, 64, 4, 210904610864373177},
	}
	for _, c := range cases {
		w := tensor.New(tensor.F64, tensor.Shape{c.out, c.in})
		ws := w.Storage().F64()
		for i := range ws {
			ws[i] = math.Sin(float64(i)*0.37) * 0.5
		}
		x := tensor.New(tensor.F64, tensor.Shape{c.in, c.samples})
		xs := x.Storage().F64()
		for i := range xs {
			xs[i] = math.Cos(float64(i)*0.11) + 0.01*float64(i%7)
		}
		q, err := nn.GPTQuantize(w, x, nn.UniformQuantizer(c.levels, -1, 1))
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		for _, v := range q.Storage().F64() {
			b := math.Float64bits(v)
			for s := 0; s < 64; s += 8 {
				h = (h ^ (b>>s)&0xff) * 1099511628211
			}
		}
		if h != c.want {
			t.Fatalf("%dx%d levels=%d digest %d, want %d", c.out, c.in, c.levels, h, c.want)
		}
	}
}
