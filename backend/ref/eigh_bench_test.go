package ref_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchEigh(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(1))
	d := make([]float64, n*n)
	for i := range n {
		for j := 0; j <= i; j++ {
			v := rng.NormFloat64()
			d[i*n+j], d[j*n+i] = v, v
		}
	}
	a := tensor.FromFloat64(tensor.Shape{n, n}, d)
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpEigh, []*tensor.Tensor{a}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEigh_16(b *testing.B) { benchEigh(b, 16) }
func BenchmarkEigh_64(b *testing.B) { benchEigh(b, 64) }
