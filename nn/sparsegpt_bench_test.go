package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchSparseGPTPrune times SparseGPT pruning, whose per-block mask selection sorts
// every column of the block once PER OUTPUT ROW — the comparator is the hot loop.
func benchSparseGPTPrune(b *testing.B, out, in, samples int) {
	w := tensor.New(tensor.F64, tensor.Shape{out, in})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Sin(float64(i)*0.37) * 0.5
	}
	// SparseGPT wants x as [in, samples] — the layer INPUT activations, not a batch.
	x := tensor.New(tensor.F64, tensor.Shape{in, samples})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Cos(float64(i)*0.11) + 0.01*float64(i%7)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := nn.SparseGPTPrune(w, x, 0.5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSparseGPTPrune_128x512(b *testing.B) { benchSparseGPTPrune(b, 128, 512, 128) }
func BenchmarkSparseGPTPrune_64x256(b *testing.B)  { benchSparseGPTPrune(b, 64, 256, 64) }
