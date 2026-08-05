package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchStepOnlyLarge isolates the optimizer Step over ONE DRAM-resident matrix (dominant-param
// regime: an LLM's embedding/lm_head/FFN matrix), where the memory-bandwidth-bound update actually
// scales across cores — unlike the ~264k-param cache-resident stepOnlyFixture.
func benchStepOnlyLarge(b *testing.B, dt tensor.Dtype, newOpt func([]*tensor.Tensor) nn.Optimizer) {
	const rows, cols = 8192, 4096 // 33.5M params, ~1 GB of f64 param+state, DRAM-resident
	p := tensor.New(dt, tensor.Shape{rows, cols})
	g := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	gd := g.Storage().F64()
	for i := range gd {
		gd[i] = float64(i%17) * 1e-3
	}
	gc := g.Cast(dt)
	gfn := func(*tensor.Tensor) *tensor.Tensor { return gc }
	opt := newOpt([]*tensor.Tensor{p})
	b.ReportAllocs()
	for b.Loop() {
		if err := opt.Step(gfn); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAdamStepOnlyLarge(b *testing.B) {
	benchStepOnlyLarge(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewAdam(p, 1e-3) })
}

func BenchmarkAdamStepOnlyLargeF32(b *testing.B) {
	benchStepOnlyLarge(b, tensor.F32, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewAdam(p, 1e-3) })
}
