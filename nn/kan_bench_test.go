package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
)

func benchKAN(b *testing.B, batch, inDim, outDim int) {
	rng := rand.New(rand.NewPCG(1, 2))
	l, err := nn.NewKAN(inDim, outDim, 1)
	if err != nil {
		b.Fatal(err)
	}
	x := randMat(rng, batch, inDim)
	ctx := backend.NewContext() // inference (Recorder == nil): the fused spline contraction
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := l.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKAN_128x128(b *testing.B) { benchKAN(b, 128, 128, 128) }
func BenchmarkKAN_256x256(b *testing.B) { benchKAN(b, 256, 256, 256) }
