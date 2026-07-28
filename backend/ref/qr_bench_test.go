package ref_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func benchQR(b *testing.B, m, n int) {
	a := bench.RandF64(tensor.Shape{m, n}, 1)
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpQR, []*tensor.Tensor{a}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQR_128x64(b *testing.B) { benchQR(b, 128, 64) }
func BenchmarkQR_32x16(b *testing.B)  { benchQR(b, 32, 16) }
