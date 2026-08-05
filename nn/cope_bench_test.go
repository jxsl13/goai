package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func benchCoPE(b *testing.B, seq, dim, heads, maxPos int) {
	rng := rand.New(rand.NewPCG(1, 2))
	c, err := nn.NewCoPEAttention(tensor.F64, dim, heads, 1, nn.WithCoPEMaxPos(maxPos))
	if err != nil {
		b.Fatal(err)
	}
	x := randMat(rng, seq, dim)
	ctx := backend.NewContext() // inference (Recorder == nil): the fused gather path
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := c.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCoPE_256x256_h4(b *testing.B) { benchCoPE(b, 256, 256, 4, 32) }
func BenchmarkCoPE_512x256_h4(b *testing.B) { benchCoPE(b, 512, 256, 4, 32) }

// Training-path CoPE forward (Recorder != nil → the differentiable position-bias gather).
func benchCoPETrain(b *testing.B, seq, dim, heads, maxPos int) {
	rng := rand.New(rand.NewPCG(1, 2))
	c, err := nn.NewCoPEAttention(tensor.F64, dim, heads, 1, nn.WithCoPEMaxPos(maxPos))
	if err != nil {
		b.Fatal(err)
	}
	x := randMat(rng, seq, dim)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		ctx := autograd.NewTape().Context() // Recorder != nil → training gather path
		if _, err := c.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCoPETrain_512x256_h4(b *testing.B)   { benchCoPETrain(b, 512, 256, 4, 32) }
func BenchmarkCoPETrain_512x256_mp64(b *testing.B) { benchCoPETrain(b, 512, 256, 4, 64) }
