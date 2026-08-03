package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchSVDVJP times the SVD backward. The forward outputs are built directly — the rule reads only
// U, s, V and their cotangents — with U and V orthonormal from a cosine basis.
//
// The singular values are kept DISTINCT and away from zero on purpose: the rule divides by
// (s_j² − s_i²) and by s_j, so a repeated or tiny value would benchmark a numerically different
// path from the one real callers take.
//
// m is strictly greater than n in both cells, because the (I−UUᵀ) correction term only runs for a
// tall matrix — benchmarking a square one would skip the most expensive loop in the rule entirely.
func benchSVDVJP(b *testing.B, m, n int) {
	vjp := vjpsMulti[backend.OpSVD]
	if vjp == nil {
		b.Fatal("no multi-output VJP registered for OpSVD")
	}
	mk := func(rows, cols int, f func(i, j int) float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		for i := range rows {
			for j := range cols {
				x.SetF64(f(i, j), i, j)
			}
		}
		return x
	}
	basis := func(rows int) func(i, j int) float64 {
		return func(i, j int) float64 {
			return math.Sqrt(2/float64(rows)) * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(rows))
		}
	}
	u := mk(m, n, basis(m))
	v := mk(n, n, basis(n))
	s := tensor.New(tensor.F64, tensor.Shape{n})
	sbar := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		s.SetF64(float64(n-i)+0.5, i) // distinct, descending, nonzero
		sbar.SetF64(math.Sin(float64(i)*0.6)*0.4, i)
	}
	ubar := mk(m, n, func(i, j int) float64 { return math.Cos(float64(i*5+j*7)) * 0.3 })
	vbar := mk(n, n, func(i, j int) float64 { return math.Sin(float64(i*11+j*3)) * 0.2 })
	outs := []*tensor.Tensor{u, s, v}
	gouts := []*tensor.Tensor{ubar, sbar, vbar}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, nil, outs, nil, gouts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSVDVJP_128x64(b *testing.B)  { benchSVDVJP(b, 128, 64) }
func BenchmarkSVDVJP_256x128(b *testing.B) { benchSVDVJP(b, 256, 128) }
