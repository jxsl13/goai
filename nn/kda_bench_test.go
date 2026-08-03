package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func benchKDA(b *testing.B, dt tensor.Dtype, seq, dk, dv int) {
	mk := func(fn func(i int) float64, r, c int) *tensor.Tensor {
		t := tensor.New(dt, tensor.Shape{r, c})
		switch dt {
		case tensor.F64:
			s := t.Storage().F64()
			for i := range s {
				s[i] = fn(i)
			}
		case tensor.F32:
			s := t.Storage().F32()
			for i := range s {
				s[i] = float32(fn(i))
			}
		}
		return t
	}
	q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, seq, dk)
	k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) }, seq, dk)
	v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) }, seq, dv)
	a := mk(func(i int) float64 { return 0.9 + 0.05*math.Cos(float64(i)*0.02) }, seq, dk)
	beta := mk(func(i int) float64 { return 0.5 }, seq, 1)
	b.ResetTimer()
	for range b.N {
		if _, err := nn.KimiDeltaAttention(q, k, v, a, beta); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKDA_F64_256x128(b *testing.B) { benchKDA(b, tensor.F64, 256, 128, 128) }
func BenchmarkKDA_F64_512x64(b *testing.B)  { benchKDA(b, tensor.F64, 512, 64, 64) }
