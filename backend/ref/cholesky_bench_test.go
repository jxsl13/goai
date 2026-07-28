package ref_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchChol(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(1))
	d := make([]float64, n*n)
	for i := range n { // symmetric positive definite
		for j := 0; j <= i; j++ {
			v := rng.NormFloat64() * 0.1
			d[i*n+j], d[j*n+i] = v, v
		}
		d[i*n+i] += float64(n)
	}
	a := tensor.FromFloat64(tensor.Shape{n, n}, d)
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpCholesky, []*tensor.Tensor{a}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCholesky_16(b *testing.B)  { benchChol(b, 16) }
func BenchmarkCholesky_128(b *testing.B) { benchChol(b, 128) }
