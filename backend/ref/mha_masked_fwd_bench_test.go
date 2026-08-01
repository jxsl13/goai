package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchMHAMaskedFwd(b *testing.B, dt tensor.Dtype, seq, dm, heads int) {
	mk := func(s float64) *tensor.Tensor {
		x := tensor.New(dt, tensor.Shape{seq, dm})
		for i := 0; i < seq; i++ {
			for j := 0; j < dm; j++ {
				x.SetF64(math.Sin(float64(i*dm+j)*0.01+s), i, j)
			}
		}
		return x
	}
	mask := tensor.New(dt, tensor.Shape{seq, seq})
	for i := 0; i < seq; i++ {
		for j := i + 1; j < seq; j++ {
			mask.SetF64(math.Inf(-1), i, j)
		}
	}
	in := []*tensor.Tensor{mk(0), mk(1), mk(2), mask}
	attrs := backend.AttnAttrs{Heads: heads}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHAMasked, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkMHAMaskedForward_F32_256h8(b *testing.B) { benchMHAMaskedFwd(b, tensor.F32, 256, 512, 8) }
func BenchmarkMHAMaskedForward_F32_512h8(b *testing.B) { benchMHAMaskedFwd(b, tensor.F32, 512, 512, 8) }
