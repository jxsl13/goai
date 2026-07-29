package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSolveSPDVJP covers the SPD-solve adjoint (Ā = −½(B̄Xᵀ+XB̄ᵀ), the O(n²k)
// contraction). n=192 SPD system with k=64 right-hand sides, F64.
func BenchmarkSolveSPDVJP(b *testing.B) {
	const n, k = 192, 64
	// A = M·Mᵀ + n·I  (symmetric positive definite).
	a := tensor.New(tensor.F64, tensor.Shape{n, n})
	as := a.Storage().F64()
	mrow := make([][]float64, n)
	for i := range mrow {
		r := make([]float64, n)
		for j := range r {
			r[j] = math.Sin(float64(i*n+j) * 0.001)
		}
		mrow[i] = r
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for t := 0; t < n; t++ {
				s += mrow[i][t] * mrow[j][t]
			}
			if i == j {
				s += float64(n)
			}
			as[i*n+j] = s
		}
	}
	rhs := tensor.New(tensor.F64, tensor.Shape{n, k})
	rs := rhs.Storage().F64()
	for i := range rs {
		rs[i] = math.Cos(float64(i) * 0.003)
	}
	ctx := backend.NewContext()
	xOut, err := backend.Execute(ctx, backend.OpSolveSPD, []*tensor.Tensor{a, rhs}, nil)
	if err != nil {
		b.Fatal(err)
	}
	x := xOut[0]
	g := tensor.New(tensor.F64, tensor.Shape{n, k})
	gs := g.Storage().F64()
	for i := range gs {
		gs[i] = math.Sin(float64(i) * 0.005)
	}
	fn := vjps[backend.OpSolveSPD]
	in := []*tensor.Tensor{a, rhs}
	out := []*tensor.Tensor{x}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := fn(ctx, in, out, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}
