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

// §V16 tier-1: singular values match numpy.linalg.svd exactly; U,V up to per-column
// sign; U·diag(s)·Vᵀ reconstructs A with orthonormal U,V columns. Goldens: numpy 2.5.1.
func TestSVDVsNumpy(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{3, 2, 2, 3, 1, 1})
	u, s, v, err := ops.SVD(a)
	if err != nil {
		t.Fatalf("SVD: %v", err)
	}
	wantS := []float64{5.196152422706632, 0.9999999999999996}
	wantU := []float64{
		-0.6804138174397718, 0.7071067811865476,
		-0.6804138174397718, -0.7071067811865474,
		-0.27216552697590873, 1.5360641181813876e-16}
	wantV := []float64{-0.7071067811865475, 0.7071067811865477, -0.7071067811865477, -0.7071067811865475}
	for j := range 2 {
		if math.Abs(s.AtF64(j)-wantS[j]) > 1e-10 {
			t.Errorf("s[%d] = %.15g, want %.15g", j, s.AtF64(j), wantS[j])
		}
		// sign-align column j of U (and V) to numpy via U's first row.
		sign := 1.0
		if u.AtF64(0, j)*wantU[j] < 0 {
			sign = -1
		}
		for i := range 3 {
			if got := sign * u.AtF64(i, j); math.Abs(got-wantU[i*2+j]) > 1e-9 {
				t.Errorf("U[%d,%d] = %.12g, want %.12g", i, j, got, wantU[i*2+j])
			}
		}
		for i := range 2 {
			if got := sign * v.AtF64(i, j); math.Abs(got-wantV[i*2+j]) > 1e-9 {
				t.Errorf("V[%d,%d] = %.12g, want %.12g", i, j, got, wantV[i*2+j])
			}
		}
	}
}

// defining properties on a tall matrix: UᵀU = Iₙ, VᵀV = Iₙ, s descending ≥ 0, U·diag(s)·Vᵀ = A.
func TestSVDProperties(t *testing.T) {
	m, n := 5, 3
	raw := []float64{
		0.5, -1.2, 0.3, 1.1, 0.2, -0.7, -0.8, 0.4, 1.5,
		0.7, -0.9, 0.2, 0.3, 1.4, -0.5,
	}
	a := tensor.FromFloat64(tensor.Shape{m, n}, raw)
	u, s, v, err := ops.SVD(a)
	if err != nil {
		t.Fatalf("SVD: %v", err)
	}
	for j := 1; j < n; j++ {
		if s.AtF64(j) > s.AtF64(j-1)+1e-12 || s.AtF64(j) < 0 {
			t.Errorf("s not descending≥0: s[%d]=%g s[%d]=%g", j-1, s.AtF64(j-1), j, s.AtF64(j))
		}
	}
	orthCols := func(name string, mat *tensor.Tensor, rows int) {
		for i := range n {
			for j := range n {
				var d float64
				for k := range rows {
					d += mat.AtF64(k, i) * mat.AtF64(k, j)
				}
				want := 0.0
				if i == j {
					want = 1
				}
				if math.Abs(d-want) > 1e-9 {
					t.Errorf("(%sᵀ%s)[%d,%d] = %g, want %g", name, name, i, j, d, want)
				}
			}
		}
	}
	orthCols("U", u, m)
	orthCols("V", v, n)
	for i := range m {
		for j := range n {
			var r float64
			for k := range n {
				r += u.AtF64(i, k) * s.AtF64(k) * v.AtF64(j, k)
			}
			if math.Abs(r-a.AtF64(i, j)) > 1e-9 {
				t.Errorf("recon[%d,%d] = %.10g, want %.10g", i, j, r, a.AtF64(i, j))
			}
		}
	}
}

func TestSVDErrors(t *testing.T) {
	if _, _, _, err := ops.SVD(tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Error("wide matrix: want error")
	}
	if _, _, _, err := ops.SVD(tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})); err == nil {
		t.Error("rank-1: want error")
	}
}

func ExampleSVD() {
	// diag(3,4) scaled: singular values are the absolute row scalings, descending.
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{3, 0, 0, 4})
	_, s, _, _ := ops.SVD(a)
	fmt.Printf("%.0f %.0f\n", s.AtF64(0), s.AtF64(1))
	// Output:
	// 4 3
}
