package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
)

func benchGatedDeltaNet(b *testing.B, seq, dk, dv int) {
	rng := rand.New(rand.NewPCG(1, 2))
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	_, alpha := randBeta(rng, seq)
	_, beta := randBeta(rng, seq)
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.GatedDeltaNet(ctx, q, k, v, alpha, beta); err != nil {
			b.Fatal(err)
		}
	}
}

func benchDeltaNet(b *testing.B, seq, dk, dv int) {
	rng := rand.New(rand.NewPCG(1, 2))
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	_, beta := randBeta(rng, seq)
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.DeltaNet(ctx, q, k, v, beta); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGatedDeltaNet_512x128(b *testing.B) { benchGatedDeltaNet(b, 512, 128, 128) }
func BenchmarkGatedDeltaNet_256x64(b *testing.B)  { benchGatedDeltaNet(b, 256, 64, 64) }
func BenchmarkDeltaNet_512x128(b *testing.B)      { benchDeltaNet(b, 512, 128, 128) }
func BenchmarkDeltaNet_256x64(b *testing.B)       { benchDeltaNet(b, 256, 64, 64) }
