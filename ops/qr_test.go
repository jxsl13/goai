package ops_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: Q,R match numpy.linalg.qr (mode='reduced') on a fixed matrix, up to
// the inherent per-column sign freedom of QR (Q[:,j],R[j,:] may be negated together).
// We align each column's sign by R[j,j] before comparing. Goldens from numpy 2.5.1.
func TestQRVsNumpy(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{12, -51, 4, 6, 167, -68, -4, 24, -41})
	q, r, err := ops.QR(a)
	if err != nil {
		t.Fatalf("QR: %v", err)
	}
	wantQ := []float64{
		-0.8571428571428572, 0.3942857142857143, 0.33142857142857146,
		-0.4285714285714286, -0.9028571428571428, -0.034285714285714246,
		0.28571428571428575, -0.1714285714285714, 0.9428571428571428}
	wantR := []float64{-14, -21, 14.000000000000002, 0, -175.00000000000003, 70, 0, 0, -35}
	// s_j: the per-column sign that maps our factorization to numpy's (Q[:,j]→s·Q[:,j],
	// R[j,:]→s·R[j,:]), determined from the diagonal R[j,j].
	for j := range 3 {
		s := 1.0
		if r.AtF64(j, j)*wantR[j*3+j] < 0 {
			s = -1
		}
		for i := range 3 {
			if got := s * q.AtF64(i, j); math.Abs(got-wantQ[i*3+j]) > 1e-12 { // column j of Q
				t.Errorf("Q[%d,%d] = %.15g, want %.15g", i, j, got, wantQ[i*3+j])
			}
			if got := s * r.AtF64(j, i); math.Abs(got-wantR[j*3+i]) > 1e-9 { // row j of R
				t.Errorf("R[%d,%d] = %.15g, want %.15g", j, i, got, wantR[j*3+i])
			}
		}
	}
}

// defining properties on a tall matrix: QᵀQ = Iₙ, R upper-triangular, Q·R = A.
func TestQRProperties(t *testing.T) {
	m, n := 5, 3
	raw := []float64{
		0.5, -1.2, 0.3, 1.1, 0.2, -0.7, -0.8, 0.4, 1.5,
		0.7, -0.9, 0.2, 0.3, 1.4, -0.5,
	}
	a := tensor.FromFloat64(tensor.Shape{m, n}, raw)
	q, r, err := ops.QR(a)
	if err != nil {
		t.Fatalf("QR: %v", err)
	}
	// QᵀQ = Iₙ
	for i := range n {
		for j := range n {
			var s float64
			for k := range m {
				s += q.AtF64(k, i) * q.AtF64(k, j)
			}
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(s-want) > 1e-10 {
				t.Errorf("(QᵀQ)[%d,%d] = %g, want %g", i, j, s, want)
			}
		}
	}
	// R upper-triangular
	for i := range n {
		for j := range i {
			if r.AtF64(i, j) != 0 {
				t.Errorf("R[%d,%d] = %g, want 0 (lower triangle)", i, j, r.AtF64(i, j))
			}
		}
	}
	// Q·R = A
	for i := range m {
		for j := range n {
			var s float64
			for k := range n {
				s += q.AtF64(i, k) * r.AtF64(k, j)
			}
			if math.Abs(s-a.AtF64(i, j)) > 1e-10 {
				t.Errorf("(QR)[%d,%d] = %.12g, want %.12g", i, j, s, a.AtF64(i, j))
			}
		}
	}
}

func TestQRErrors(t *testing.T) {
	// wide matrix (m < n) is unsupported
	if _, _, err := ops.QR(tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Error("wide matrix: want error")
	}
	// non-rank-2
	if _, _, err := ops.QR(tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})); err == nil {
		t.Error("rank-1: want error")
	}
}

func ExampleQR() {
	a := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	q, r, _ := ops.QR(a)
	// QR guarantees orthonormal columns and reconstruction, regardless of column signs.
	var qtq00, recon00 float64
	for k := range 3 {
		qtq00 += q.AtF64(k, 0) * q.AtF64(k, 0)
	}
	for k := range 2 {
		recon00 += q.AtF64(0, k) * r.AtF64(k, 0)
	}
	fmt.Printf("QᵀQ[0,0]=%.4f  (QR)[0,0]=%.4f  R lower=%.1f\n", qtq00, recon00, r.AtF64(1, 0))
	// Output:
	// QᵀQ[0,0]=1.0000  (QR)[0,0]=1.0000  R lower=0.0
}
