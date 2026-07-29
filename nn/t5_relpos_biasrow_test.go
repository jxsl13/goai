package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestT5BiasRowMatchesFull proves BiasRow(q, k) is bit-identical to the q-th row of the full
// Bias(q+1, k) — the exact substitution the incremental T5 decoder relies on.
func TestT5BiasRowMatchesFull(t *testing.T) {
	const nb, nh = 32, 4
	for _, bidir := range []bool{false, true} {
		b, _ := nn.NewT5RelativeBias(nb, nh, 128, bidir, tensor.F64)
		for k := 0; k < nb; k++ {
			for h := 0; h < nh; h++ {
				b.Table.SetF64(float64(k)*0.37-float64(h)*1.1, k, h)
			}
		}
		ctx := backend.NewContext()
		for _, pos := range []int{0, 1, 5, 17, 40} {
			kk := pos + 1
			full, err := b.Bias(ctx, pos+1, kk)
			if err != nil {
				t.Fatal(err)
			}
			row, err := b.BiasRow(ctx, pos, kk)
			if err != nil {
				t.Fatal(err)
			}
			if !row.Shape().Equal(tensor.Shape{kk, nh}) {
				t.Fatalf("BiasRow shape %v, want [%d %d]", row.Shape(), kk, nh)
			}
			for k := 0; k < kk; k++ {
				for h := 0; h < nh; h++ {
					if got, want := row.AtF64(k, h), full.AtF64(pos, k, h); got != want {
						t.Fatalf("bidir=%v pos=%d BiasRow[%d][%d]=%g != Bias[%d][%d][%d]=%g",
							bidir, pos, k, h, got, pos, k, h, want)
					}
				}
			}
		}
	}
}

// benchmarks: the incremental-decode bias row at a large position — the old full-table-then-slice
// path vs BiasRow. selfBiasRow (llamagpu) calls this once per generated token.
const biasRowPos = 512

func BenchmarkT5BiasRow_FullThenSlice(b *testing.B) {
	bias, _ := nn.NewT5RelativeBias(32, 8, 128, false, tensor.F64)
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		full, err := bias.Bias(ctx, biasRowPos+1, biasRowPos+1)
		if err != nil {
			b.Fatal(err)
		}
		_ = full.AtF64(biasRowPos, 0, 0) // touch the used row
	}
}

func BenchmarkT5BiasRow_Row(b *testing.B) {
	bias, _ := nn.NewT5RelativeBias(32, 8, 128, false, tensor.F64)
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := bias.BiasRow(ctx, biasRowPos, biasRowPos+1); err != nil {
			b.Fatal(err)
		}
	}
}
