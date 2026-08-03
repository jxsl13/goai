package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchTranspose measures the path every default-backend caller takes. OpTranspose had no cpu
// benchmark at all, which is how a serial kernel on the per-head attention path went unseen.
func benchTranspose(b *testing.B, m, n int, ref bool) {
	x := tensor.New(tensor.F32, tensor.Shape{m, n})
	xs := x.Storage().F32()
	for i := range xs {
		xs[i] = float32(math.Sin(float64(i)))
	}
	ctx := backend.NewContext()
	if ref {
		ctx = ctx.WithBackend(backend.Reference())
	}
	in := []*tensor.Tensor{x}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpTranspose, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTransposeCPU_512x64(b *testing.B)   { benchTranspose(b, 512, 64, false) }
func BenchmarkTransposeRef_512x64(b *testing.B)   { benchTranspose(b, 512, 64, true) }
func BenchmarkTransposeCPU_1024x768(b *testing.B) { benchTranspose(b, 1024, 768, false) }
func BenchmarkTransposeRef_1024x768(b *testing.B) { benchTranspose(b, 1024, 768, true) }
