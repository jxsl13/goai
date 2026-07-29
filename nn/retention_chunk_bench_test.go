package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkRetentionChunkwise covers the RetNet chunkwise (prefill) retention form.
// L=512, dk=dv=64, chunk=64.
func BenchmarkRetentionChunkwise(b *testing.B) {
	const L, dk, dv = 512, 64, 64
	mk := func(fn func(i int) float64, c int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{L, c})
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, dk)
	k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) }, dk)
	v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) }, dv)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.RetentionChunkwise(q, k, v, 0.968, 64); err != nil {
			b.Fatal(err)
		}
	}
}
