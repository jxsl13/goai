package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchWKV(b *testing.B, dt tensor.Dtype, seq, d int) {
	mk := func(r, c int, s float64) *tensor.Tensor {
		x := tensor.New(dt, tensor.Shape{r, c})
		st := x.Storage()
		for i := 0; i < r*c; i++ {
			v := math.Sin(float64(i)*0.01+s) * 0.5
			if dt == tensor.F64 {
				st.F64()[i] = v
			} else {
				st.F32()[i] = float32(v)
			}
		}
		return x
	}
	vec := func(n int, s float64) *tensor.Tensor {
		x := tensor.New(dt, tensor.Shape{n})
		for i := 0; i < n; i++ {
			x.SetF64(0.3+0.1*math.Cos(float64(i)*0.03+s), i)
		}
		return x
	}
	in := []*tensor.Tensor{mk(seq, d, 0.1), mk(seq, d, 0.2), vec(d, 0.3), vec(d, 0.4)}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpWKV, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkWKV_F32_512x2048(b *testing.B) { benchWKV(b, tensor.F32, 512, 2048) }
func BenchmarkWKV_F32_1024x1024(b *testing.B) { benchWKV(b, tensor.F32, 1024, 1024) }
