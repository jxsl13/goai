package ref_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func benchSVD(b *testing.B, m, n int) {
	a := bench.RandF64(tensor.Shape{m, n}, 1)
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpSVD, []*tensor.Tensor{a}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSVD_64x32(b *testing.B) { benchSVD(b, 64, 32) }
func BenchmarkSVD_16x8(b *testing.B)  { benchSVD(b, 16, 8) }
