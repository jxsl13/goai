package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkRetentionRecurrent covers the RetNet recurrent (decode) form. The output
// dot out_n = Q_n·S_n reads the [dk,dv] state; the loop-interchange makes that read
// contiguous. Larger dv widens the state so the strided-vs-contiguous gap shows.
func benchRetentionRecurrent(b *testing.B, L, dk, dv int) {
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
		if _, err := nn.RetentionRecurrent(q, k, v, 0.968); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRetentionRecurrent_256x256(b *testing.B) { benchRetentionRecurrent(b, 256, 256, 256) }
func BenchmarkRetentionRecurrent_512x128(b *testing.B) { benchRetentionRecurrent(b, 512, 128, 128) }
func BenchmarkRetentionRecurrent_256x64(b *testing.B)  { benchRetentionRecurrent(b, 256, 64, 64) }
