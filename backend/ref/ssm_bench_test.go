package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchSSM(b *testing.B, dt tensor.Dtype, L, D, N int) {
	mk := func(shape tensor.Shape, s float64) *tensor.Tensor {
		x := tensor.New(dt, shape)
		st := x.Storage()
		for i := 0; i < x.Numel(); i++ {
			v := math.Sin(float64(i)*0.01+s) * 0.3
			if dt == tensor.F64 {
				st.F64()[i] = v
			} else {
				st.F32()[i] = float32(v)
			}
		}
		return x
	}
	in := []*tensor.Tensor{
		mk(tensor.Shape{L, D}, 0), mk(tensor.Shape{L, D}, 1),
		mk(tensor.Shape{D, N}, 2), mk(tensor.Shape{L, N}, 3), mk(tensor.Shape{L, N}, 4),
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpSSM, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkSSM_F32_512x2048x16(b *testing.B)  { benchSSM(b, tensor.F32, 512, 2048, 16) }
func BenchmarkSSM_F32_1024x1024x16(b *testing.B) { benchSSM(b, tensor.F32, 1024, 1024, 16) }
