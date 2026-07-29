package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func benchHGRNSeq(b *testing.B, l, d int) {
	m, err := nn.NewHGRN(tensor.F64, d, nn.WithHGRNSeed(1), nn.WithHGRNLowerBound(0.3))
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := tensor.New(tensor.F64, tensor.Shape{l, d})
	for i := range l * d {
		x.SetF64(rng.NormFloat64(), tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := m.ForwardSequential(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHGRNSeq_512x512(b *testing.B) { benchHGRNSeq(b, 512, 512) }
func BenchmarkHGRNSeq_256x256(b *testing.B) { benchHGRNSeq(b, 256, 256) }
