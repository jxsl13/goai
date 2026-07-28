package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// qrRows is the ORIGINAL row-of-slices Householder QR, kept as the oracle for the
// flat rewrite: identical operations, operands and order, only the buffer layout
// differs, so the two must agree BIT for bit. A property check (Q·R == A, Qᵀ·Q == I)
// would NOT do — it tolerates exactly the drift being guarded against, and the
// package suite proved unable to catch either of two deliberate index mutations.
func qrRows(ad []float64, m, n int) (q, r [][]float64) {
	rm := make([][]float64, m)
	for i := range m {
		rm[i] = make([]float64, n)
		for j := range n {
			rm[i][j] = ad[i*n+j]
		}
	}
	vs := make([][]float64, n)
	betas := make([]float64, n)
	for k := range n {
		var nrm float64
		for i := k; i < m; i++ {
			nrm += rm[i][k] * rm[i][k]
		}
		nrm = math.Sqrt(nrm)
		v := make([]float64, m)
		vs[k] = v
		if nrm == 0 {
			continue
		}
		alpha := -nrm
		if rm[k][k] < 0 {
			alpha = nrm
		}
		for i := k; i < m; i++ {
			v[i] = rm[i][k]
		}
		v[k] -= alpha
		var vtv float64
		for i := k; i < m; i++ {
			vtv += v[i] * v[i]
		}
		if vtv == 0 {
			continue
		}
		beta := 2 / vtv
		betas[k] = beta
		for j := k; j < n; j++ {
			var s float64
			for i := k; i < m; i++ {
				s += v[i] * rm[i][j]
			}
			bs := beta * s
			for i := k; i < m; i++ {
				rm[i][j] -= bs * v[i]
			}
		}
		rm[k][k] = alpha
		for i := k + 1; i < m; i++ {
			rm[i][k] = 0
		}
	}
	qq := make([][]float64, m)
	for i := range m {
		qq[i] = make([]float64, n)
		if i < n {
			qq[i][i] = 1
		}
	}
	for k := n - 1; k >= 0; k-- {
		if betas[k] == 0 {
			continue
		}
		v := vs[k]
		for j := range n {
			var s float64
			for i := k; i < m; i++ {
				s += v[i] * qq[i][j]
			}
			bs := betas[k] * s
			for i := k; i < m; i++ {
				qq[i][j] -= bs * v[i]
			}
		}
	}
	return qq, rm
}

func TestQRFlatBitIdenticalToRows(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, sz := range [][2]int{{1, 1}, {3, 2}, {8, 8}, {17, 5}, {32, 16}} {
		m, n := sz[0], sz[1]
		a := bench.RandF64(tensor.Shape{m, n}, uint64(m*100+n))
		ad := make([]float64, m*n)
		for i := range m {
			for j := range n {
				ad[i*n+j] = a.AtF64(i, j)
			}
		}
		out, err := backend.Execute(ctx, backend.OpQR, []*tensor.Tensor{a}, nil)
		if err != nil {
			t.Fatal(err)
		}
		wantQ, wantR := qrRows(ad, m, n)
		for i := range m {
			for j := range n {
				got := out[0].AtF64(i, j)
				if math.Float64bits(got) != math.Float64bits(wantQ[i][j]) {
					t.Fatalf("%dx%d Q[%d,%d]: got %v want %v", m, n, i, j, got, wantQ[i][j])
				}
			}
		}
		for i := range n {
			for j := i; j < n; j++ {
				got := out[1].AtF64(i, j)
				if math.Float64bits(got) != math.Float64bits(wantR[i][j]) {
					t.Fatalf("%dx%d R[%d,%d]: got %v want %v", m, n, i, j, got, wantR[i][j])
				}
			}
		}
	}
}
