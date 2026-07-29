package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSmoothQuant covers SmoothQuant activation/weight smoothing (absmax scan +
// X̂/Ŵ rescale). tokens=512, C_in=1024, C_out=1024.
func BenchmarkSmoothQuant(b *testing.B) {
	const tokens, cin, cout = 512, 1024, 1024
	x := tensor.New(tensor.F64, tensor.Shape{tokens, cin})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Sin(float64(i) * 0.017)
	}
	w := tensor.New(tensor.F64, tensor.Shape{cin, cout})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Cos(float64(i) * 0.013)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, _, err := nn.SmoothQuant(x, w, 0.5); err != nil {
			b.Fatal(err)
		}
	}
}
