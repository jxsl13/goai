package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The existing GaLore benchmark uses 64x64 and 64x128 parameters, whose Gram matrices are
// cache-resident. PROC-BENCH-CACHE-THRESHOLD-001 requires a second size whose working set
// leaves cache, since a memory-access rewrite measures as noise below that threshold.
func benchGaLoreStep(b *testing.B, shapes []tensor.Shape) {
	params := make([]*tensor.Tensor, len(shapes))
	grads := map[*tensor.Tensor]*tensor.Tensor{}
	for i, s := range shapes {
		params[i] = tensor.New(tensor.F64, s)
		g := tensor.New(tensor.F64, s)
		gd := g.Storage().F64()
		for j := range gd {
			gd[j] = float64(j%17) * 1e-3
		}
		grads[params[i]] = g
	}
	gf := func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
	opt := nn.NewGaLore(params, 1e-3)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := opt.Step(gf); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGaLoreStepLarge(b *testing.B) {
	benchGaLoreStep(b, []tensor.Shape{{256, 512}, {512}})
}
func BenchmarkGaLoreStepWide(b *testing.B) {
	benchGaLoreStep(b, []tensor.Shape{{512, 256}, {256}})
}
