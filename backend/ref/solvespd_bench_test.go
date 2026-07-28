package ref_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchSolveSPD(b *testing.B, n, k int) {
	rng := rand.New(rand.NewSource(1))
	ad := make([]float64, n*n)
	for i := range n {
		for j := 0; j <= i; j++ {
			v := rng.NormFloat64() * 0.1
			ad[i*n+j], ad[j*n+i] = v, v
		}
		ad[i*n+i] += float64(n) + 1
	}
	bd := make([]float64, n*k)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	a := tensor.FromFloat64(tensor.Shape{n, n}, ad)
	rhs := tensor.FromFloat64(tensor.Shape{n, k}, bd)
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpSolveSPD, []*tensor.Tensor{a, rhs}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSolveSPD_64x16(b *testing.B) { benchSolveSPD(b, 64, 16) }
func BenchmarkSolveSPD_16x4(b *testing.B)  { benchSolveSPD(b, 16, 4) }
