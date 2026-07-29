package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// BenchmarkVQQuantize covers VQ-VAE codebook assignment (argmin over K codebook rows) at a
// typical VQGAN scale: batch=512 latents, D=256, K=1024 codebook. The nearest-codebook scan
// dominates.
func BenchmarkVQQuantize(b *testing.B) {
	const batch, d, k = 512, 256, 1024
	vq := nn.NewVectorQuantizer(tensor.F64, k, d, 0.25)
	cb := vq.Codebook.Storage().F64()
	for i := range cb {
		cb[i] = math.Sin(float64(i) * 0.017)
	}
	ze := tensor.New(tensor.F64, tensor.Shape{batch, d})
	zs := ze.Storage().F64()
	for i := range zs {
		zs[i] = math.Cos(float64(i) * 0.013)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, _, err := vq.Quantize(ctx, ze); err != nil {
			b.Fatal(err)
		}
	}
}
