package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkGSPO covers the GSPO sequence-policy loss over a training batch: G=64 sequences
// of 256 tokens each (16384 per-token log-probs).
func BenchmarkGSPO(b *testing.B) {
	const g, seqLen = 64, 256
	total := g * seqLen
	lengths := make([]int, g)
	for i := range lengths {
		lengths[i] = seqLen
	}
	mk := func(n int, fn func(i int) float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{n})
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	lpNew := mk(total, func(i int) float64 { return -1 + 0.01*math.Sin(float64(i)*0.1) })
	lpOld := mk(total, func(i int) float64 { return -1 + 0.01*math.Cos(float64(i)*0.1) })
	adv := mk(g, func(i int) float64 { return math.Sin(float64(i)) })
	ctx := backend.NewContext()
	attrs := backend.GSPOAttrs{Epsilon: 3e-4, Lengths: lengths}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpGSPO, []*tensor.Tensor{lpNew, lpOld, adv}, attrs); err != nil {
			b.Fatal(err)
		}
	}
}
