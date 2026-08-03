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

// BenchmarkSVD_256x128 is the cell where a layout change can show. The existing 64x32 and 16x8 cells
// hold their whole working set in L1 — 16 KB and 1 KB — so what the Jacobi sweep walks does not
// matter there, and a real transform reads as noise (§PS6011, and the QR pair that measured -35.0%
// at 128x64 and nothing at 32x16). This one is 256 KB of column data, past L1 and into L2.
func BenchmarkSVD_256x128(b *testing.B) { benchSVD(b, 256, 128) }
