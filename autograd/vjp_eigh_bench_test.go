package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchEighVJP times the symmetric-eigendecomposition backward. It needs the FORWARD outputs —
// eigenvalues w and eigenvectors V — which are built here directly rather than by factoring: the
// rule reads nothing else, and V only has to be orthonormal enough that the arithmetic is
// representative. A rotation-composed basis gives that exactly.
//
// The eigenvalues are kept well separated. That is not cosmetic: the rule divides by (w_j − w_i),
// so a near-degenerate spectrum would put the benchmark on a numerically different path from the
// one real callers take.
func benchEighVJP(b *testing.B, n int) {
	vjp := vjpsMulti[backend.OpEigh]
	if vjp == nil {
		b.Fatal("no multi-output VJP registered for OpEigh")
	}
	w := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		w.SetF64(1+0.5*float64(i), i) // ascending, spaced
	}
	v := tensor.New(tensor.F64, tensor.Shape{n, n})
	for i := range n {
		for j := range n {
			// A discrete cosine basis: orthonormal columns, no degenerate structure.
			v.SetF64(math.Sqrt(2/float64(n))*math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(n)), i, j)
		}
	}
	wbar := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		wbar.SetF64(math.Sin(float64(i)*0.7)*0.3, i)
	}
	vbar := tensor.New(tensor.F64, tensor.Shape{n, n})
	for i := range n {
		for j := range n {
			vbar.SetF64(math.Cos(float64(i*5+j*3))*0.2, i, j)
		}
	}
	outs := []*tensor.Tensor{w, v}
	gouts := []*tensor.Tensor{wbar, vbar}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, nil, outs, nil, gouts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEighVJP_64(b *testing.B)  { benchEighVJP(b, 64) }
func BenchmarkEighVJP_128(b *testing.B) { benchEighVJP(b, 128) }
func BenchmarkEighVJP_256(b *testing.B) { benchEighVJP(b, 256) }
