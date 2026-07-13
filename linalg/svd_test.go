package linalg_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// reconstructSVD builds U·diag(s)·Vᵀ from the SVD factors (U [m×p], s [p], V [n×p]).
func reconstructSVD(u, s, v *tensor.Tensor) *tensor.Tensor {
	m, p := u.Shape()[0], u.Shape()[1]
	n := v.Shape()[0]
	out := tensor.New(tensor.F64, tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			var acc float64
			for k := range p {
				acc += u.AtF64(i, k) * s.AtF64(k) * v.AtF64(j, k)
			}
			out.SetF64(acc, i, j)
		}
	}
	return out
}

// orthonormalCols checks that the columns of q [r×c] are orthonormal (qᵀq = I_c).
func orthonormalCols(t *testing.T, name string, q *tensor.Tensor, tol float64) {
	t.Helper()
	r, c := q.Shape()[0], q.Shape()[1]
	for i := range c {
		for j := range c {
			var s float64
			for k := range r {
				s += q.AtF64(k, i) * q.AtF64(k, j)
			}
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(s-want) > tol {
				t.Errorf("%s (colsᵀcols)[%d,%d] = %g, want %g", name, i, j, s, want)
			}
		}
	}
}

// §V16 tier-1: the one-sided Jacobi SVD reconstructs A = U·Σ·Vᵀ, U and V have orthonormal columns,
// and the singular values are non-negative and sorted descending (§R117). Tested over tall, square
// and wide matrices.
func TestSVDReconstruct(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 42))
	for _, d := range [][2]int{{1, 1}, {3, 3}, {6, 4}, {4, 6}, {8, 8}, {10, 3}, {2, 5}} {
		m, n := d[0], d[1]
		ad := make([]float64, m*n)
		for i := range ad {
			ad[i] = rng.NormFloat64()
		}
		a := tensor.FromFloat64(tensor.Shape{m, n}, ad)
		u, s, v, err := linalg.SVD(a)
		if err != nil {
			t.Fatalf("%dx%d: %v", m, n, err)
		}
		p := min(m, n)
		if !s.Shape().Equal(tensor.Shape{p}) || !u.Shape().Equal(tensor.Shape{m, p}) || !v.Shape().Equal(tensor.Shape{n, p}) {
			t.Fatalf("%dx%d shapes U%v s%v V%v", m, n, u.Shape(), s.Shape(), v.Shape())
		}
		// singular values non-negative and descending
		for k := range p {
			if s.AtF64(k) < -1e-12 {
				t.Errorf("%dx%d σ[%d] = %g negative", m, n, k, s.AtF64(k))
			}
			if k > 0 && s.AtF64(k) > s.AtF64(k-1)+1e-12 {
				t.Errorf("%dx%d σ not descending: σ[%d]=%g > σ[%d]=%g", m, n, k, s.AtF64(k), k-1, s.AtF64(k-1))
			}
		}
		orthonormalCols(t, fmt.Sprintf("%dx%d U", m, n), u, 1e-9)
		orthonormalCols(t, fmt.Sprintf("%dx%d V", m, n), v, 1e-9)
		// reconstruction
		rec := reconstructSVD(u, s, v)
		for i := range m {
			for j := range n {
				if math.Abs(rec.AtF64(i, j)-a.AtF64(i, j)) > 1e-9 {
					t.Errorf("%dx%d (UΣVᵀ−A)[%d,%d] = %g", m, n, i, j, rec.AtF64(i, j)-a.AtF64(i, j))
				}
			}
		}
	}
}

// §V16 tier-1: for a diagonal matrix the singular values are the sorted absolute diagonal entries.
func TestSVDDiagonal(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{2, 0, 0, 0, -5, 0, 0, 0, 3})
	_, s, _, _ := linalg.SVD(a)
	want := []float64{5, 3, 2}
	for k, w := range want {
		if math.Abs(s.AtF64(k)-w) > 1e-12 {
			t.Errorf("σ[%d] = %g, want %g", k, s.AtF64(k), w)
		}
	}
}

// §V16 tier-1: the largest singular value equals the spectral (2-)norm; here the singular values of
// a known 2×2 are checked, and σ₁ dominates.
func TestSVDKnownAndRankDeficient(t *testing.T) {
	// rank-1 matrix [[1,2],[2,4]] = [1,2]ᵀ·[1,2] → σ₁=5, σ₂=0
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 4})
	_, s, _, _ := linalg.SVD(a)
	if math.Abs(s.AtF64(0)-5) > 1e-9 || math.Abs(s.AtF64(1)) > 1e-9 {
		t.Errorf("rank-1: σ = [%g,%g], want [5,0]", s.AtF64(0), s.AtF64(1))
	}
}

func TestSVDErrors(t *testing.T) {
	if _, _, _, err := linalg.SVD(tensor.New(tensor.F64, tensor.Shape{3})); err == nil {
		t.Error("non-matrix must error")
	}
	if _, _, _, err := linalg.SVD(tensor.New(tensor.F64, tensor.Shape{0, 2})); err == nil {
		t.Error("empty matrix must error")
	}
}

func ExampleSVD() {
	// diagonal matrix: singular values are the sorted |diagonal|
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{3, 0, 0, -4})
	_, s, _, _ := linalg.SVD(a)
	fmt.Printf("%.0f %.0f\n", s.AtF64(0), s.AtF64(1))
	// Output: 4 3
}
