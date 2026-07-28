package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchCholeskyVJP times the Cholesky backward. It needs the FORWARD factor L as out[0]
// — a lower-triangular matrix — which is built here directly rather than by factoring,
// since the VJP never inspects anything but L's lower triangle.
func benchCholeskyVJP(b *testing.B, n int) {
	vjp := vjps[backend.OpCholesky]
	if vjp == nil {
		b.Fatal("no VJP registered for OpCholesky")
	}
	l := tensor.New(tensor.F64, tensor.Shape{n, n})
	g := tensor.New(tensor.F64, tensor.Shape{n, n})
	for i := range n {
		for j := 0; j <= i; j++ {
			v := math.Cos(float64(i*7+j*3)) * 0.4
			if i == j {
				v = 1.5 + 0.1*float64(i%5) // keep the diagonal well away from zero
			}
			l.SetF64(v, i, j)
			g.SetF64(math.Sin(float64(i*5+j*11))*0.3, i, j)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, nil, []*tensor.Tensor{l}, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCholeskyVJP_64(b *testing.B)  { benchCholeskyVJP(b, 64) }
func BenchmarkCholeskyVJP_128(b *testing.B) { benchCholeskyVJP(b, 128) }
