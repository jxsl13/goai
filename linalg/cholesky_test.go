package linalg_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// spd builds a random symmetric positive-definite matrix A = B·Bᵀ + n·I.
func spd(rng *rand.Rand, n int) *tensor.Tensor {
	b := randRect(rng, n, n)
	a := tensor.New(tensor.F64, tensor.Shape{n, n})
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += b.AtF64(i, k) * b.AtF64(j, k) // (B·Bᵀ)[i,j]
			}
			if i == j {
				s += float64(n)
			}
			a.SetF64(s, i, j)
		}
	}
	return a
}

// §V16 tier-1: Cholesky reconstructs A = L·Lᵀ, L is lower-triangular with positive diagonal (§R120).
func TestCholeskyReconstruct(t *testing.T) {
	rng := rand.New(rand.NewPCG(51, 99))
	for _, n := range []int{1, 2, 3, 5, 8} {
		a := spd(rng, n)
		l, err := linalg.Cholesky(a)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		for i := range n {
			if l.AtF64(i, i) <= 0 {
				t.Errorf("n=%d L[%d,%d] = %g, want > 0", n, i, i, l.AtF64(i, i))
			}
			for j := i + 1; j < n; j++ {
				if l.AtF64(i, j) != 0 {
					t.Errorf("n=%d L[%d,%d] = %g (not lower-triangular)", n, i, j, l.AtF64(i, j))
				}
			}
		}
		// L·Lᵀ == A
		for i := range n {
			for j := range n {
				var s float64
				for k := range n {
					s += l.AtF64(i, k) * l.AtF64(j, k)
				}
				if math.Abs(s-a.AtF64(i, j)) > 1e-9 {
					t.Errorf("n=%d (L·Lᵀ−A)[%d,%d] = %g", n, i, j, s-a.AtF64(i, j))
				}
			}
		}
	}
}

// §V16 tier-1: the SPD solve gives A·x = b, matching the general LU solve.
func TestCholSolve(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 17))
	const n = 6
	a := spd(rng, n)
	bd := make([]float64, n)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	b := tensor.FromFloat64(tensor.Shape{n}, bd)
	x, err := linalg.CholSolve(a, b)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		var s float64
		for j := range n {
			s += a.AtF64(i, j) * x.AtF64(j)
		}
		if math.Abs(s-b.AtF64(i)) > 1e-9 {
			t.Errorf("residual[%d] = %g", i, s-b.AtF64(i))
		}
	}
	// matches the LU solve
	lu, _ := linalg.Solve(a, b)
	for i := range n {
		if math.Abs(x.AtF64(i)-lu.AtF64(i)) > 1e-8 {
			t.Errorf("CholSolve x[%d]=%g != LU %g", i, x.AtF64(i), lu.AtF64(i))
		}
	}
	// matrix RHS
	bm := tensor.FromFloat64(tensor.Shape{n, 2}, make([]float64, n*2))
	for i := range n {
		bm.SetF64(rng.NormFloat64(), i, 0)
		bm.SetF64(rng.NormFloat64(), i, 1)
	}
	xm, err := linalg.CholSolve(a, bm)
	if err != nil || !xm.Shape().Equal(tensor.Shape{n, 2}) {
		t.Fatalf("matrix RHS: %v shape %v", err, xm.Shape())
	}
}

// §V16 tier-1: a symmetric but indefinite/non-SPD matrix is rejected.
func TestCholeskyNonSPD(t *testing.T) {
	// [[1,2],[2,1]] is symmetric with eigenvalues 3 and −1 → not positive definite
	indef := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 1})
	if _, err := linalg.Cholesky(indef); err == nil {
		t.Error("indefinite matrix must be rejected as not positive definite")
	}
	// negative diagonal
	if _, err := linalg.Cholesky(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{-1, 0, 0, 1})); err == nil {
		t.Error("non-positive-definite matrix must be rejected")
	}
}

func TestCholeskyErrors(t *testing.T) {
	if _, err := linalg.Cholesky(tensor.New(tensor.F64, tensor.Shape{2, 3})); err == nil {
		t.Error("non-square must error")
	}
	if _, err := linalg.Cholesky(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})); err == nil {
		t.Error("non-symmetric must error")
	}
}

func ExampleCholesky() {
	// A = [[4,2],[2,3]] → L = [[2,0],[1,√2]]
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{4, 2, 2, 3})
	l, _ := linalg.Cholesky(a)
	fmt.Printf("%.4f %.4f %.4f\n", l.AtF64(0, 0), l.AtF64(1, 0), l.AtF64(1, 1))
	// Output: 2.0000 1.0000 1.4142
}
