package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchMoEBalance(b *testing.B, dt tensor.Dtype, tks, n int) {
	x := tensor.New(dt, tensor.Shape{tks, n})
	st := x.Storage()
	for i := 0; i < tks*n; i++ {
		v := math.Sin(float64(i)*0.01) * 2
		if dt == tensor.F64 {
			st.F64()[i] = v
		} else {
			st.F32()[i] = float32(v)
		}
	}
	as := tensor.New(tensor.F64, tensor.Shape{tks})
	for t := 0; t < tks; t++ {
		as.SetF64(float64(t%n), t)
	}
	in := []*tensor.Tensor{x, as}
	attrs := backend.MoEBalanceAttrs{Alpha: 0.01}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMoEBalance, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkMoEBalance_F32_16384x128(b *testing.B) { benchMoEBalance(b, tensor.F32, 16384, 128) }
func BenchmarkMoEBalance_F64_16384x128(b *testing.B) { benchMoEBalance(b, tensor.F64, 16384, 128) }
