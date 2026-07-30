package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchDistill(b *testing.B, dt tensor.Dtype, batch, classes int) {
	mk := func(s float64) *tensor.Tensor {
		x := tensor.New(dt, tensor.Shape{batch, classes})
		st := x.Storage()
		for i := 0; i < batch*classes; i++ {
			v := math.Sin(float64(i)*0.001+s) * 2
			if dt == tensor.F64 {
				st.F64()[i] = v
			} else {
				st.F32()[i] = float32(v)
			}
		}
		return x
	}
	in := []*tensor.Tensor{mk(0), mk(1)}
	attrs := backend.DistillAttrs{Temperature: 2.0}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpDistill, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkDistill_F32_512x32000(b *testing.B) { benchDistill(b, tensor.F32, 512, 32000) }
func BenchmarkDistill_F32_256x50000(b *testing.B) { benchDistill(b, tensor.F32, 256, 50000) }
