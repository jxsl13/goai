package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkUniformSoup measures uniform model-soup averaging of 5 fine-tunes over a
// critic-sized model (~131k params). Both the cross-model accumulation and the rescaled
// write walked every element with Unravel + AtF64/SetF64; the typed contiguous fast path
// reads each contiguous model straight from its backing []float64.
func BenchmarkUniformSoup(b *testing.B) {
	const M = 5
	shapes := []tensor.Shape{{256, 256}, {256, 256}, {256, 1}}
	models := make([][]*tensor.Tensor, M)
	for t := range models {
		models[t] = make([]*tensor.Tensor, len(shapes))
		for i, sh := range shapes {
			models[t][i] = tensor.New(tensor.F64, sh)
			f := models[t][i].Storage().F64()
			for e := range f {
				f[e] = 0.01 * float64((t+i+e)%17-8)
			}
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := nn.UniformSoup(models); err != nil {
			b.Fatal(err)
		}
	}
}
