package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchQRVJP times the QR backward. It builds the forward outputs directly — the rule reads only Q,
// R and their cotangents — with Q orthonormal from a cosine basis and R upper-triangular with a
// diagonal well away from zero, since the rule inverts R by back-substitution.
//
// Sizes are chosen to cross L1 (§SIZE-THE-CELL-PAST-L1-BEFORE-JUDGING-LAYOUT): 128x64 is about
// 64 KB of Q, 256x128 about 256 KB. A layout change measured only at the smaller size reads as a
// fraction of what it is worth at the larger one.
func benchQRVJP(b *testing.B, m, n int) {
	vjp := vjpsMulti[backend.OpQR]
	if vjp == nil {
		b.Fatal("no multi-output VJP registered for OpQR")
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
	q := mk(m, n, func(i, j int) float64 {
		return math.Sqrt(2/float64(m)) * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(m))
	})
	r := mk(n, n, func(i, j int) float64 {
		if i > j {
			return 0
		}
		if i == j {
			return 2 + 0.5*float64(i%7)
		}
		return math.Sin(float64(i*3+j*5)) * 0.4
	})
	qbar := mk(m, n, func(i, j int) float64 { return math.Cos(float64(i*7+j*3)) * 0.3 })
	rbar := mk(n, n, func(i, j int) float64 {
		if i > j {
			return 0
		}
		return math.Sin(float64(i*5+j*11)) * 0.25
	})
	outs := []*tensor.Tensor{q, r}
	gouts := []*tensor.Tensor{qbar, rbar}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, nil, outs, nil, gouts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQRVJP_128x64(b *testing.B)  { benchQRVJP(b, 128, 64) }
func BenchmarkQRVJP_256x128(b *testing.B) { benchQRVJP(b, 256, 128) }
