package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMoBAAttention covers Mixture-of-Block-Attention. seq=512, dm=512, heads=8,
// blockSize=64, topK=4.
func BenchmarkMoBAAttention(b *testing.B) {
	const seq, dm, heads = 512, 512, 8
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
	v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) })
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.MoBAAttention(q, k, v, heads, 64, 4, 0); err != nil {
			b.Fatal(err)
		}
	}
}
