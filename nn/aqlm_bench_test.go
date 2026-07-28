package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchEncodeAQLM times AQLM codebook learning — the k-means passes and the codebook
// refit, which is where the per-element row indexing sits.
func benchEncodeAQLM(b *testing.B, rows, cols int) {
	w := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Sin(float64(i)*0.13) * 0.6
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := nn.EncodeAQLM(w); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodeAQLM_256x256(b *testing.B) { benchEncodeAQLM(b, 256, 256) }
func BenchmarkEncodeAQLM_128x512(b *testing.B) { benchEncodeAQLM(b, 128, 512) }
