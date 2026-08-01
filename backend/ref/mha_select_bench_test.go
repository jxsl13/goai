package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchMHASelect(b *testing.B, dt tensor.Dtype, seq, dm, heads int) {
	mk := func(r, c int, s float64) *tensor.Tensor {
		x := tensor.New(dt, tensor.Shape{r, c})
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				x.SetF64(math.Sin(float64(i*c+j)*0.01+s), i, j)
			}
		}
		return x
	}
	sel := mk(seq, seq, 5)
	in := []*tensor.Tensor{mk(seq, dm, 0), mk(seq, dm, 1), mk(seq, dm, 2), mk(seq, dm, 3), mk(seq, dm, 4), sel}
	attrs := backend.AttnAttrs{Heads: heads}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHASelect, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkMHASelect_F32_256h8(b *testing.B) { benchMHASelect(b, tensor.F32, 256, 512, 8) }
func BenchmarkMHASelect_F32_512h8(b *testing.B) { benchMHASelect(b, tensor.F32, 512, 512, 8) }
