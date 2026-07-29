package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkWandaPrune covers Wanda unstructured pruning (per-output-column score sort).
// C_in=1024, C_out=512, tokens=128, 50% sparsity.
func BenchmarkWandaPrune(b *testing.B) {
	const cin, cout, tokens = 1024, 512, 128
	w := tensor.New(tensor.F64, tensor.Shape{cin, cout})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Sin(float64(i) * 0.017)
	}
	x := tensor.New(tensor.F64, tensor.Shape{tokens, cin})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Cos(float64(i) * 0.011)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := nn.WandaPrune(w, x, 0.5); err != nil {
			b.Fatal(err)
		}
	}
}
