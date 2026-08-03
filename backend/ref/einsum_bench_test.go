package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchEinsum(b *testing.B, spec string, shapes ...tensor.Shape) {
	ins := make([]*tensor.Tensor, len(shapes))
	for k, sh := range shapes {
		x := tensor.New(tensor.F64, sh)
		st := x.Storage().F64()
		for i := range st {
			st[i] = math.Sin(float64(i)*0.01 + float64(k))
		}
		ins[k] = x
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpEinsum, ins, backend.EinsumAttrs{Spec: spec}); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkEinsum_KAN_128(b *testing.B) {
	benchEinsum(b, "bic,ijc->bij", tensor.Shape{128, 128, 8}, tensor.Shape{128, 128, 8})
}
func BenchmarkEinsum_bmm(b *testing.B) {
	benchEinsum(b, "trh,trd->thd", tensor.Shape{256, 32, 8}, tensor.Shape{256, 32, 64})
}
