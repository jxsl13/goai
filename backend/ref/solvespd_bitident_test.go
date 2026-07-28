package ref_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// solveSPDRows is the ORIGINAL row-of-slices, accessor-driven substitution, kept as
// the oracle. Written BEFORE the rewrite it guards, because a one-ulp mutation in
// the forward substitution passed the entire backend/ref and autograd suites — this
// kernel had no correctness coverage at the level the rewrite touches, exactly as QR
// did not (bcf9e13).
func solveSPDRows(lf []float64, bd []float64, n, k int) [][]float64 {
	at := func(i, j int) float64 { return lf[i*n+j] }
	x := make([][]float64, n)
	y := make([][]float64, n)
	for i := range n {
		x[i] = make([]float64, k)
		y[i] = make([]float64, k)
	}
	for c := range k {
		for i := range n {
			s := bd[i*k+c]
			for p := range i {
				s -= at(i, p) * y[p][c]
			}
			y[i][c] = s / at(i, i)
		}
		for i := n - 1; i >= 0; i-- {
			s := y[i][c]
			for p := i + 1; p < n; p++ {
				s -= at(p, i) * x[p][c]
			}
			x[i][c] = s / at(i, i)
		}
	}
	return x
}

func TestSolveSPDBitIdentical(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, sz := range [][2]int{{1, 1}, {2, 1}, {3, 4}, {9, 2}, {16, 8}} {
		n, k := sz[0], sz[1]
		rng := rand.New(rand.NewSource(int64(n*100 + k)))
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
		b := tensor.FromFloat64(tensor.Shape{n, k}, bd)
		out, err := backend.Execute(ctx, backend.OpSolveSPD, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		// The oracle needs L; take it from the same Cholesky the kernel uses.
		lout, err := backend.Execute(ctx, backend.OpCholesky, []*tensor.Tensor{a}, nil)
		if err != nil {
			t.Fatal(err)
		}
		lf := make([]float64, n*n)
		for i := range n {
			for j := range n {
				lf[i*n+j] = lout[0].AtF64(i, j)
			}
		}
		want := solveSPDRows(lf, bd, n, k)
		for i := range n {
			for c := range k {
				got := out[0].AtF64(i, c)
				if math.Float64bits(got) != math.Float64bits(want[i][c]) {
					t.Fatalf("n=%d k=%d x[%d,%d]: got %v want %v", n, k, i, c, got, want[i][c])
				}
			}
		}
	}
}
