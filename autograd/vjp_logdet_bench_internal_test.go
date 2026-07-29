package autograd

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// spdMatrix builds a well-conditioned SPD matrix A = (1/n)·M·Mᵀ + n·I.
func spdMatrix(n int, rng *rand.Rand) *tensor.Tensor {
	m := make([]float64, n*n)
	for i := range m {
		m[i] = rng.NormFloat64()
	}
	a := tensor.New(tensor.F64, tensor.Shape{n, n})
	as := a.Storage().F64()
	inv := 1.0 / float64(n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += m[i*n+k] * m[j*n+k]
			}
			as[i*n+j] = s * inv
		}
		as[i*n+i] += float64(n)
	}
	return a
}

func benchLogDetBackward(b *testing.B, n int) {
	vjp := vjps[backend.OpLogDet]
	rng := rand.New(rand.NewPCG(11, 0x9e3779b9))
	a := spdMatrix(n, rng)
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 1.0
	ctx := backend.NewContext().WithBackend(backend.Reference())
	in := []*tensor.Tensor{a}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := vjp(ctx, in, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLogDetBackward_256(b *testing.B) { benchLogDetBackward(b, 256) }
func BenchmarkLogDetBackward_512(b *testing.B) { benchLogDetBackward(b, 512) }
