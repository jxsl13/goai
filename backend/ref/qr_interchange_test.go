package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestQRInterchangeIsBitIdentical locks the column-innermost reflector application to the
// arithmetic it replaced. The reference below is the original nest — one column at a time, each
// accumulating down a strided column — so the two differ in exactly one respect: the order in which
// independent columns are visited.
//
// Exact bits, both outputs. Shapes include a square case, a tall one, and n=1 where the trailing
// loop is empty.
func TestQRInterchangeIsBitIdentical(t *testing.T) {
	for _, sz := range [][2]int{{1, 1}, {4, 1}, {5, 5}, {9, 4}, {32, 16}, {65, 33}} {
		m, n := sz[0], sz[1]
		a := tensor.New(tensor.F64, tensor.Shape{m, n})
		src := make([]float64, m*n)
		for i := range m {
			for j := range n {
				v := math.Sin(float64(i*13+j*7))*2 + 0.3*float64((i+j)%3)
				src[i*n+j] = v
				a.SetF64(v, i, j)
			}
		}
		got, err := backend.Execute(backend.NewContext(), backend.OpQR, []*tensor.Tensor{a}, nil)
		if err != nil {
			t.Fatalf("%dx%d: %v", m, n, err)
		}
		wantQ, wantR := qrColumnOuter(m, n, src)
		for i := range m {
			for j := range n {
				if q := got[0].AtF64(i, j); math.Float64bits(q) != math.Float64bits(wantQ[i*n+j]) {
					t.Fatalf("%dx%d Q[%d,%d]: interchanged %v, column-outer %v — not bit-identical",
						m, n, i, j, q, wantQ[i*n+j])
				}
			}
		}
		for i := range n {
			for j := range n {
				if r := got[1].AtF64(i, j); math.Float64bits(r) != math.Float64bits(wantR[i*n+j]) {
					t.Fatalf("%dx%d R[%d,%d]: interchanged %v, column-outer %v — not bit-identical",
						m, n, i, j, r, wantR[i*n+j])
				}
			}
		}
	}
}

// qrColumnOuter is the pre-interchange Householder factorization, kept verbatim.
func qrColumnOuter(m, n int, src []float64) (q, r []float64) {
	rm := make([]float64, m*n)
	copy(rm, src)
	vs := make([][]float64, n)
	betas := make([]float64, n)
	for k := range n {
		var nrm float64
		for i := k; i < m; i++ {
			nrm += rm[i*n+k] * rm[i*n+k]
		}
		nrm = math.Sqrt(nrm)
		v := make([]float64, m)
		vs[k] = v
		if nrm == 0 {
			continue
		}
		alpha := -nrm
		if rm[k*n+k] < 0 {
			alpha = nrm
		}
		for i := k; i < m; i++ {
			v[i] = rm[i*n+k]
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
				s += v[i] * rm[i*n+j]
			}
			bs := beta * s
			for i := k; i < m; i++ {
				rm[i*n+j] -= bs * v[i]
			}
		}
		rm[k*n+k] = alpha
		for i := k + 1; i < m; i++ {
			rm[i*n+k] = 0
		}
	}
	q = make([]float64, m*n)
	for i := range m {
		if i < n {
			q[i*n+i] = 1
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
				s += v[i] * q[i*n+j]
			}
			bs := betas[k] * s
			for i := k; i < m; i++ {
				q[i*n+j] -= bs * v[i]
			}
		}
	}
	return q, rm
}
