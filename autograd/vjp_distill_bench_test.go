package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkDistillVJP covers the knowledge-distillation gradient T·(q−p) over a
// realistic student/teacher logit batch [batch, vocab] = [128, 8192], F64.
func BenchmarkDistillVJP(b *testing.B) {
	const bs, c = 128, 8192
	mk := func(seed int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{bs, c})
		s := t.Storage().F64()
		for i := range s {
			s[i] = math.Sin(float64((i+seed)*3) * 0.0007)
		}
		return t
	}
	zs, zt := mk(0), mk(500)
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.SetF64(1.0)
	ctx := backend.NewContext()
	fn := vjps[backend.OpDistill]
	in := []*tensor.Tensor{zs, zt}
	attrs := backend.DistillAttrs{Temperature: 2.0}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := fn(ctx, in, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}
