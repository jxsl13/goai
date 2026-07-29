package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkLLMInt8MatMul covers the LLM.int8() mixed-precision matmul; the int8 path over
// the regular (non-outlier) columns dominates. tokens=256, cin=cout=512, threshold high so
// the int8 path carries the work.
func BenchmarkLLMInt8MatMul(b *testing.B) {
	const tokens, cin, cout = 256, 512, 512
	x := tensor.New(tensor.F64, tensor.Shape{tokens, cin})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Sin(float64(i) * 0.017) // bounded, no outliers
	}
	w := tensor.New(tensor.F64, tensor.Shape{cin, cout})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Cos(float64(i) * 0.013)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := nn.LLMInt8MatMul(x, w, 6.0); err != nil {
			b.Fatal(err)
		}
	}
}
