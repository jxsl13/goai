package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func BenchmarkMoETrain_512x512_e8k2(b *testing.B) {
	m := nn.NewSparseMoE(tensor.F64, 512, 1024, 8, 2, 1)
	rng := rand.New(rand.NewPCG(1, 2))
	x := tensor.New(tensor.F64, tensor.Shape{512, 512})
	for i := 0; i < x.Numel(); i++ {
		x.SetF64(rng.NormFloat64()*0.1, tensor.Unravel(i, x.Shape())...)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		ctx := autograd.NewTape().Context() // Recorder != nil → training path
		if _, _, err := m.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}
