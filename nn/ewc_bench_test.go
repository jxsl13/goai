package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkEWCFisher measures the Fisher-information estimate over a critic-sized
// MLP (~131k params) averaged across 8 gradient samples — the once-per-task cost
// of Elastic Weight Consolidation. The typed contiguous fast path replaces the
// per-element Unravel + AtF64 + SetF64 dispatch the generic walk paid on every
// (sample × parameter) with a direct []float64 sum-of-squares.
func BenchmarkEWCFisher(b *testing.B) {
	const nS = 8
	shapes := []tensor.Shape{{256, 256}, {256, 256}, {256, 1}}
	gradSamples := make([][]*tensor.Tensor, nS)
	for s := range gradSamples {
		gradSamples[s] = make([]*tensor.Tensor, len(shapes))
		for i, sh := range shapes {
			t := tensor.New(tensor.F64, sh)
			f := t.Storage().F64()
			for e := range f {
				f[e] = 0.001 * float64((s+i+e)%13)
			}
			gradSamples[s][i] = t
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := nn.EWCFisher(gradSamples); err != nil {
			b.Fatal(err)
		}
	}
}
