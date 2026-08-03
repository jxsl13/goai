package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestLLMInt8MatMulIsBitIdentical freezes the dequantized product. Banding the token rows
// claims to change no value — each output still accumulates over the same channels in the same
// order into the same four partials — and a quantized matmul is somewhere a small drift would
// be attributed to the quantization and never investigated.
//
// One shape is deliberately below the fan-out gate and two clear it, and one carries OUTLIER
// channels so the mixed path — where some columns are skipped in the quantized product and
// handled separately — is covered rather than only the clean one.
func TestLLMInt8MatMulIsBitIdentical(t *testing.T) {
	cases := []struct {
		tokens, cin, cout int
		outliers          bool
		want              uint64
	}{
		{4, 8, 8, false, 8678963366648527939},
		{96, 128, 64, false, 4467307680355772428},
		{64, 96, 96, true, 17148586048111164472},
	}
	for _, c := range cases {
		x := tensor.New(tensor.F64, tensor.Shape{c.tokens, c.cin})
		xs := x.Storage().F64()
		for i := range xs {
			xs[i] = math.Sin(float64(i) * 0.017)
		}
		if c.outliers {
			for i := 0; i < len(xs); i += c.cin {
				xs[i+c.cin/3] = 40 // one channel far above the threshold, in every row
			}
		}
		w := tensor.New(tensor.F64, tensor.Shape{c.cin, c.cout})
		ws := w.Storage().F64()
		for i := range ws {
			ws[i] = math.Cos(float64(i) * 0.013)
		}
		y, idx, err := nn.LLMInt8MatMul(x, w, 6.0)
		if err != nil {
			t.Fatal(err)
		}
		if c.outliers && len(idx) == 0 {
			t.Fatal("want at least one outlier channel, got none — the mixed path is untested")
		}
		h := uint64(14695981039346656037)
		for _, v := range y.Storage().F64() {
			b := math.Float64bits(v)
			for s := 0; s < 64; s += 8 {
				h = (h ^ (b>>s)&0xff) * 1099511628211
			}
		}
		if h != c.want {
			t.Fatalf("%dx%dx%d outliers=%v digest %d, want %d",
				c.tokens, c.cin, c.cout, c.outliers, h, c.want)
		}
	}
}
