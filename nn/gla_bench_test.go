package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
)

func benchGLA(b *testing.B, seq, dk, dv int) {
	rng := rand.New(rand.NewPCG(1, 2))
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	gate := randGate(rng, seq, dk)
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.GatedLinearAttention(ctx, q, k, v, gate); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGLA_512x128(b *testing.B) { benchGLA(b, 512, 128, 128) }
func BenchmarkGLA_256x64(b *testing.B)  { benchGLA(b, 256, 64, 64) }
