package cpu

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// benchDistillF32On measures F32 distill on a chosen backend — PRE (Ref, serial) vs POST
// (CPU, row-parallel) A/B for the dtype-gap win.
func benchDistillF32On(b *testing.B, name backend.Name, batch, c int) {
	be, _ := backend.Get(name)
	in := []*tensor.Tensor{
		bench.RandF32(tensor.Shape{batch, c}, 1),
		bench.RandF32(tensor.Shape{batch, c}, 2),
	}
	attrs := backend.DistillAttrs{Temperature: 2.0}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpDistill, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDistillF32Ref_64x8000(b *testing.B)   { benchDistillF32On(b, backend.Ref, 64, 8000) }
func BenchmarkDistillF32CPU_64x8000(b *testing.B)   { benchDistillF32On(b, backend.CPU, 64, 8000) }
func BenchmarkDistillF32Ref_256x32000(b *testing.B) { benchDistillF32On(b, backend.Ref, 256, 32000) }
func BenchmarkDistillF32CPU_256x32000(b *testing.B) { benchDistillF32On(b, backend.CPU, 256, 32000) }
