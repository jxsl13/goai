package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSIConsolidate measures the Synaptic-Intelligence task-boundary consolidation
// (Ω += ω/((Δθ)²+ξ), advance the anchor) over a critic-sized MLP (~131k params). The
// omega/bigOmega math already runs on flat []float64 masters; the typed contiguous fast
// path removes the remaining per-element AtF64(Unravel) read of each parameter. NewSI in
// setup also exercises the typed snapshot() path (3 full-parameter copies).
func BenchmarkSIConsolidate(b *testing.B) {
	shapes := []tensor.Shape{{256, 256}, {256, 256}, {256, 1}}
	params := make([]*tensor.Tensor, len(shapes))
	for i, sh := range shapes {
		params[i] = tensor.New(tensor.F64, sh)
		f := params[i].Storage().F64()
		for e := range f {
			f[e] = 0.01 * float64((i+e)%11-5)
		}
	}
	si := nn.NewSI(params)
	b.ReportAllocs()
	for b.Loop() {
		si.Consolidate(0.1)
	}
}
