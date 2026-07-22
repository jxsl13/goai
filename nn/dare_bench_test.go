package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkDARE measures DARE model merging (drop-and-rescale of the task vector) over a
// critic-sized model (~131k params). Every element drew a drop decision and did an
// Unravel + AtF64 (base) + AtF64 (model) + SetF64; the typed contiguous fast path keeps
// the per-element rng draw (so the survivor mask is unchanged) but walks the backing
// []float64 directly instead of dispatching.
func BenchmarkDARE(b *testing.B) {
	shapes := []tensor.Shape{{256, 256}, {256, 256}, {256, 1}}
	base := make([]*tensor.Tensor, len(shapes))
	model := make([]*tensor.Tensor, len(shapes))
	for i, sh := range shapes {
		base[i] = tensor.New(tensor.F64, sh)
		model[i] = tensor.New(tensor.F64, sh)
		bf, mf := base[i].Storage().F64(), model[i].Storage().F64()
		for e := range bf {
			bf[e] = 0.01 * float64((i+e)%17-8)
			mf[e] = 0.01 * float64((i+e)%11-5)
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := nn.DARE(base, model, 0.5, 42); err != nil {
			b.Fatal(err)
		}
	}
}
