package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func benchSinkhorn(b *testing.B, m, n, iters int) {
	cd := make([]float64, m*n)
	for i := range cd {
		cd[i] = float64((i*7)%101) * 0.01
	}
	cost := tensor.FromFloat64(tensor.Shape{m, n}, cd)
	r := make([]float64, m)
	for i := range r {
		r[i] = 1.0 / float64(m)
	}
	c := make([]float64, n)
	for j := range c {
		c[j] = 1.0 / float64(n)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := nn.Sinkhorn(cost, r, c, 0.05, iters); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSinkhorn1024x1024i20(b *testing.B) { benchSinkhorn(b, 1024, 1024, 20) }
func BenchmarkSinkhorn512x512i3(b *testing.B)    { benchSinkhorn(b, 512, 512, 3) }
