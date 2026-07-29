package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMaxAttentionLogits covers Kimi K2 QK-Clip's max-logit scan (O(heads·seq²·dk)).
// seq=256, dm=256, heads=4, causal.
func BenchmarkMaxAttentionLogits(b *testing.B) {
	const seq, dm, heads = 256, 256, 4
	mk := func(fn func(i int) float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{seq, dm})
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) })
	k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) })
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.MaxAttentionLogits(q, k, heads, 0.125, true); err != nil {
			b.Fatal(err)
		}
	}
}
