package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchArgMax(b *testing.B, rows, cols int) {
	x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	st := x.Storage().F64()
	for i := range st {
		st[i] = math.Sin(float64(i) * 0.7)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpArgMax, []*tensor.Tensor{x}, backend.ArgMaxAttrs{Axis: 1}); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkArgMax_4096x4096(b *testing.B)  { benchArgMax(b, 4096, 4096) }
func BenchmarkArgMax_2048x32000(b *testing.B) { benchArgMax(b, 2048, 32000) }
