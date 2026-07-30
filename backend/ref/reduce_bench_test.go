package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchReduce(b *testing.B, op backend.Op, rows, cols int) {
	x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	st := x.Storage().F64()
	for i := range st {
		st[i] = math.Sin(float64(i) * 0.001)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkReduceSum_4096x4096(b *testing.B) { benchReduce(b, backend.OpSum, 4096, 4096) }
func BenchmarkReduceMax_4096x4096(b *testing.B) { benchReduce(b, backend.OpMax, 4096, 4096) }
