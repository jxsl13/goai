package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchZLoss(b *testing.B, dt tensor.Dtype, batch, classes int) {
	x := tensor.New(dt, tensor.Shape{batch, classes})
	st := x.Storage()
	for i := 0; i < batch*classes; i++ {
		v := math.Sin(float64(i)*0.001) * 3
		if dt == tensor.F64 {
			st.F64()[i] = v
		} else {
			st.F32()[i] = float32(v)
		}
	}
	in := []*tensor.Tensor{x}
	attrs := backend.ZLossAttrs{Coeff: 1e-4}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpZLoss, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkZLoss_F32_512x32000(b *testing.B) { benchZLoss(b, tensor.F32, 512, 32000) }
func BenchmarkZLoss_F64_512x32000(b *testing.B) { benchZLoss(b, tensor.F64, 512, 32000) }
