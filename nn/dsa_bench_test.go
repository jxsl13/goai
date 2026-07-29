package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkDSAAttention covers DeepSeek Sparse Attention (lightning-indexer ranking +
// selected attention). seq=256, dm=256, heads=4, idxHeads=4, idxDim=64, topK=32.
func BenchmarkDSAAttention(b *testing.B) {
	const seq, dm, heads, idxHeads, idxDim = 256, 256, 4, 4, 64
	mk := func(fn func(i int) float64, c int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{seq, c})
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, dm)
	k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) }, dm)
	v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) }, dm)
	qIdx := mk(func(i int) float64 { return math.Cos(float64(i) * 0.007) }, idxHeads*idxDim)
	kIdx := mk(func(i int) float64 { return math.Sin(float64(i) * 0.009) }, idxHeads*idxDim)
	w := []float64{0.5, 0.3, 0.15, 0.05}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.DSAAttention(q, k, v, qIdx, kIdx, w, heads, 32, 0); err != nil {
			b.Fatal(err)
		}
	}
}
