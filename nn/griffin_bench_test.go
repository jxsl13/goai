package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func benchRGLRUSeq(b *testing.B, l, d int) {
	m, err := nn.NewRGLRU(tensor.F64, d, nn.WithRGLRUSeed(1))
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

func BenchmarkRGLRUSeq_512x512(b *testing.B) { benchRGLRUSeq(b, 512, 512) }
func BenchmarkRGLRUSeq_256x256(b *testing.B) { benchRGLRUSeq(b, 256, 256) }
