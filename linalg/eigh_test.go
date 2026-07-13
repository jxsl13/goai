package linalg_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

func randSym(rng *rand.Rand, n int) *tensor.Tensor {
	b := randRect(rng, n, n)
	a := tensor.New(tensor.F64, tensor.Shape{n, n})
	for i := range n {
		for j := range n {
			a.SetF64((b.AtF64(i, j)+b.AtF64(j, i))/2, i, j)
		}
	}
	return a
}

// §V16 tier-1: Eigh of a symmetric matrix satisfies the eigenvalue equation A·V[:,k] = w[k]·V[:,k],
// reconstructs A = V·diag(w)·Vᵀ, has orthonormal eigenvectors (VᵀV = I), and returns eigenvalues in
// ASCENDING order (§R119).
func TestEighReconstruct(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 82))
	for _, n := range []int{1, 2, 3, 5, 8} {
		a := randSym(rng, n)
		w, v, err := linalg.Eigh(a)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		if !w.Shape().Equal(tensor.Shape{n}) || !v.Shape().Equal(tensor.Shape{n, n}) {
			t.Fatalf("n=%d shapes w%v V%v", n, w.Shape(), v.Shape())
		}
		// ascending eigenvalues
		for k := 1; k < n; k++ {
			if w.AtF64(k) < w.AtF64(k-1)-1e-12 {
				t.Errorf("n=%d eigenvalues not ascending: w[%d]=%g < w[%d]=%g", n, k, w.AtF64(k), k-1, w.AtF64(k-1))
			}
		}
		// VᵀV = I (orthonormal eigenvectors, columns)
		orthonormalCols(t, fmt.Sprintf("n=%d V", n), v, 1e-9)
		// A·V[:,k] = w[k]·V[:,k]
		for k := range n {
			for i := range n {
				var s float64
				for j := range n {
					s += a.AtF64(i, j) * v.AtF64(j, k)
				}
				if math.Abs(s-w.AtF64(k)*v.AtF64(i, k)) > 1e-8 {
					t.Errorf("n=%d (A·v−λv)[%d,%d] = %g", n, i, k, s-w.AtF64(k)*v.AtF64(i, k))
				}
			}
		}
		// reconstruction A = V·diag(w)·Vᵀ
		for i := range n {
			for j := range n {
				var acc float64
				for k := range n {
					acc += v.AtF64(i, k) * w.AtF64(k) * v.AtF64(j, k)
				}
				if math.Abs(acc-a.AtF64(i, j)) > 1e-8 {
					t.Errorf("n=%d (VΛVᵀ−A)[%d,%d] = %g", n, i, j, acc-a.AtF64(i, j))
				}
			}
		}
	}
}

// §V16 tier-1: for a diagonal symmetric matrix the eigenvalues are the sorted (ascending) diagonal.
func TestEighDiagonal(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{5, 0, 0, 0, -2, 0, 0, 0, 3})
	w, _, _ := linalg.Eigh(a)
	want := []float64{-2, 3, 5}
	for k, x := range want {
		if math.Abs(w.AtF64(k)-x) > 1e-12 {
			t.Errorf("w[%d] = %g, want %g", k, w.AtF64(k), x)
		}
	}
}

// §V16 tier-1: a known 2×2 [[2,1],[1,2]] has eigenvalues 1 and 3 (eigenvectors [1,−1]/√2, [1,1]/√2).
func TestEighKnown2x2(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 1, 1, 2})
	w, v, _ := linalg.Eigh(a)
	if math.Abs(w.AtF64(0)-1) > 1e-12 || math.Abs(w.AtF64(1)-3) > 1e-12 {
		t.Errorf("eigenvalues [%g,%g], want [1,3]", w.AtF64(0), w.AtF64(1))
	}
	// eigenvector for λ=3 is ∝ [1,1]: components equal in magnitude
	if math.Abs(math.Abs(v.AtF64(0, 1))-math.Abs(v.AtF64(1, 1))) > 1e-9 {
		t.Errorf("λ=3 eigenvector not ∝[1,1]: %v", []float64{v.AtF64(0, 1), v.AtF64(1, 1)})
	}
}

func TestEighErrors(t *testing.T) {
	if _, _, err := linalg.Eigh(tensor.New(tensor.F64, tensor.Shape{2, 3})); err == nil {
		t.Error("non-square must error")
	}
	// clearly non-symmetric
	if _, _, err := linalg.Eigh(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})); err == nil {
		t.Error("non-symmetric must error")
	}
}

func ExampleEigh() {
	// [[2,1],[1,2]] → eigenvalues 1 and 3 (ascending)
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 1, 1, 2})
	w, _, _ := linalg.Eigh(a)
	fmt.Printf("%.0f %.0f\n", w.AtF64(0), w.AtF64(1))
	// Output: 1 3
}
