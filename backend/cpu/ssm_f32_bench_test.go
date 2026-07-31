package cpu

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchSSMF32On measures F32 SSM on a chosen backend — PRE (Ref, serial) vs POST (CPU,
// channel-parallel) A/B for the dtype-gap win. Mamba/S6 dims: L seq, D model, N state.
func benchSSMF32On(b *testing.B, name backend.Name, L, D, N int) {
	be, _ := backend.Get(name)
	in := []*tensor.Tensor{
		ssmMkF32(tensor.Shape{L, D}, 0.1), ssmMkF32(tensor.Shape{L, D}, 0.2),
		ssmMkF32(tensor.Shape{D, N}, 0.3), ssmMkF32(tensor.Shape{L, N}, 0.4), ssmMkF32(tensor.Shape{L, N}, 0.5),
	}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpSSM, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSSMF32Ref_512x1024x16(b *testing.B)  { benchSSMF32On(b, backend.Ref, 512, 1024, 16) }
func BenchmarkSSMF32CPU_512x1024x16(b *testing.B)  { benchSSMF32On(b, backend.CPU, 512, 1024, 16) }
func BenchmarkSSMF32Ref_1024x2048x16(b *testing.B) { benchSSMF32On(b, backend.Ref, 1024, 2048, 16) }
func BenchmarkSSMF32CPU_1024x2048x16(b *testing.B) { benchSSMF32On(b, backend.CPU, 1024, 2048, 16) }
