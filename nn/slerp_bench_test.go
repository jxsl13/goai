package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSLERP measures spherical linear interpolation of two model-sized weight
// tensors (~131k elements) — the per-parameter cost of SLERP model merging. Both the
// norm/dot pass and the interpolation pass walked every element with Unravel +
// AtF64/SetF64; the typed contiguous fast path replaces that with a direct []float64
// walk.
func BenchmarkSLERP(b *testing.B) {
	const n = 131072
	a := tensor.New(tensor.F64, tensor.Shape{n})
	c := tensor.New(tensor.F64, tensor.Shape{n})
	af, cf := a.Storage().F64(), c.Storage().F64()
	for i := range af {
		af[i] = 0.01 * float64((i%17)-8)
		cf[i] = 0.01 * float64((i%13)-6)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := nn.SLERP(a, c, 0.3); err != nil {
			b.Fatal(err)
		}
	}
}
