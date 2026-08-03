package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSSDRecurrent covers the Mamba-2 SSD linear-time scan. The per-step output
// y_t = Cᵀ_t·h reads the [N,d] state; the loop-interchange makes that read contiguous.
// Larger d widens the state so the strided-vs-contiguous gap shows.
func benchSSDRecurrent(b *testing.B, T, d, n int) {
	mk := func(fn func(i int) float64, r, c int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{r, c})
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	x := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, T, d)
	a := tensor.New(tensor.F64, tensor.Shape{T}) // rank-1 scalar decays a[T]
	as := a.Storage().F64()
	for i := range as {
		as[i] = 0.9 + 0.05*math.Cos(float64(i)*0.02)
	}
	bb := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) }, T, n)
	c := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) }, T, n)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.SSDRecurrent(x, a, bb, c); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSSDRecurrent_256d256n128(b *testing.B) { benchSSDRecurrent(b, 256, 256, 128) }
func BenchmarkSSDRecurrent_512d128n64(b *testing.B)  { benchSSDRecurrent(b, 512, 128, 64) }
func BenchmarkSSDRecurrent_256d64n16(b *testing.B)   { benchSSDRecurrent(b, 256, 64, 16) }
