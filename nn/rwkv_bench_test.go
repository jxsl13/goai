package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkWKV covers the RWKV WKV attention recurrence (per-channel sequential scan),
// the core RWKV inference kernel. seq=512, D=1024.
func BenchmarkWKV(b *testing.B) {
	const seq, d = 512, 1024
	mk := func(fn func(i int) float64, shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape(shape))
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	k := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, seq, d)
	v := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) }, seq, d)
	w := mk(func(i int) float64 { return -0.5 - 0.1*math.Abs(math.Sin(float64(i))) }, d)
	u := mk(func(i int) float64 { return 0.3 * math.Cos(float64(i)) }, d)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.WKV(k, v, w, u); err != nil {
			b.Fatal(err)
		}
	}
}
