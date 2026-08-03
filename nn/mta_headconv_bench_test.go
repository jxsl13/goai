package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func benchMTAForward(b *testing.B, T, dim, heads, cq, ck, ch int) {
	m, err := nn.NewMultiTokenAttention(tensor.F32, dim, heads, 1,
		nn.WithMTAKeyQueryKernel(cq, ck), nn.WithMTAHeadKernel(ch))
	if err != nil {
		b.Fatal(err)
	}
	x := tensor.New(tensor.F32, tensor.Shape{T, dim})
	xs := x.Storage().F32()
	for i := range xs {
		xs[i] = float32(math.Sin(float64(i) * 0.001))
	}
	ctx := backend.NewContext() // inference: Recorder == nil
	b.ResetTimer()
	for range b.N {
		if _, err := m.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMTAForward_ch16(b *testing.B) { benchMTAForward(b, 512, 2048, 32, 6, 11, 16) }
func BenchmarkMTAForward_ch8(b *testing.B)  { benchMTAForward(b, 256, 1024, 16, 6, 11, 8) }
