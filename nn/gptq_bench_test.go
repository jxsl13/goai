package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func benchGPTQuantize(b *testing.B, out, in, samples int) {
	w := tensor.New(tensor.F64, tensor.Shape{out, in})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Sin(float64(i)*0.37) * 0.5
	}
	x := tensor.New(tensor.F64, tensor.Shape{in, samples}) // layer input activations
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Cos(float64(i)*0.11) + 0.01*float64(i%7)
	}
	quant := func(v float64) float64 { return math.Round(v*8) / 8 } // simple 4-bit-ish grid
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := nn.GPTQuantize(w, x, quant); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGPTQuantize_128x512(b *testing.B) { benchGPTQuantize(b, 128, 512, 128) }
func BenchmarkGPTQuantize_64x256(b *testing.B)  { benchGPTQuantize(b, 64, 256, 64) }
