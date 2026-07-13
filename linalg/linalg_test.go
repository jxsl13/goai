package linalg_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// matmul is a small independent [n×n]·[n×k] product for verifying identities.
func matmul(a, b *tensor.Tensor) *tensor.Tensor {
	n, kk := a.Shape()[0], b.Shape()[1]
	m := a.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{n, kk})
	for i := range n {
		for j := range kk {
			var s float64
			for p := range m {
				s += a.AtF64(i, p) * b.AtF64(p, j)
			}
			out.SetF64(s, i, j)
		}
	}
	return out
}

func randMatrix(rng *rand.Rand, n int) *tensor.Tensor {
	d := make([]float64, n*n)
	for i := range d {
		d[i] = rng.NormFloat64()
	}
	// add n·I to keep it well-conditioned / non-singular
	a := tensor.FromFloat64(tensor.Shape{n, n}, d)
	for i := range n {
		a.SetF64(a.AtF64(i, i)+float64(n), i, i)
	}
	return a
}

// §V16 tier-1: Solve returns x with A·x = b to within tolerance (the residual identity, §R115).
func TestSolveResidual(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{1, 2, 3, 5, 8} {
		a := randMatrix(rng, n)
		bd := make([]float64, n)
		for i := range bd {
			bd[i] = rng.NormFloat64()
		}
		b := tensor.FromFloat64(tensor.Shape{n}, bd)
		x, err := linalg.Solve(a, b)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		// residual A·x − b
		for i := range n {
			var s float64
			for j := range n {
				s += a.AtF64(i, j) * x.AtF64(j)
			}
			if math.Abs(s-b.AtF64(i)) > 1e-9 {
				t.Errorf("n=%d residual[%d] = %g", n, i, s-b.AtF64(i))
			}
		}
	}
}

// §V16 tier-1: A·A⁻¹ = I (the inverse identity).
func TestInverseIdentity(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, n := range []int{1, 2, 4, 7} {
		a := randMatrix(rng, n)
		inv, err := linalg.Inverse(a)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		prod := matmul(a, inv)
		for i := range n {
			for j := range n {
				want := 0.0
				if i == j {
					want = 1
				}
				if math.Abs(prod.AtF64(i, j)-want) > 1e-9 {
					t.Errorf("n=%d (A·A⁻¹)[%d,%d] = %g, want %g", n, i, j, prod.AtF64(i, j), want)
				}
			}
		}
	}
}

// §V16 tier-1: determinant against hand-computed small cases and the identities det(I)=1,
// det(A)·det(A⁻¹)=1, det(2×2)=ad−bc, det(triangular)=∏ diagonal.
func TestDet(t *testing.T) {
	// 2×2: [[1,2],[3,4]] → 1·4 − 2·3 = −2
	d, err := linalg.Det(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4}))
	if err != nil || math.Abs(d-(-2)) > 1e-12 {
		t.Errorf("det([[1,2],[3,4]]) = %g, want −2 (err %v)", d, err)
	}
	// 3×3 upper-triangular: diagonal product 2·3·4 = 24
	tri := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{2, 5, 1, 0, 3, 7, 0, 0, 4})
	if d, _ := linalg.Det(tri); math.Abs(d-24) > 1e-12 {
		t.Errorf("det(triangular) = %g, want 24", d)
	}
	// identity
	if d, _ := linalg.Det(tensor.Eye(tensor.F64, 5)); math.Abs(d-1) > 1e-12 {
		t.Errorf("det(I) = %g, want 1", d)
	}
	// det(A)·det(A⁻¹) = 1
	rng := rand.New(rand.NewPCG(5, 6))
	a := randMatrix(rng, 6)
	da, _ := linalg.Det(a)
	inv, _ := linalg.Inverse(a)
	di, _ := linalg.Det(inv)
	if math.Abs(da*di-1) > 1e-8 {
		t.Errorf("det(A)·det(A⁻¹) = %g, want 1", da*di)
	}
}

// §V16 tier-1: Solve with a matrix right-hand side [n,k] solves each column; A·X = B.
func TestSolveMultipleRHS(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const n, k = 5, 3
	a := randMatrix(rng, n)
	bd := make([]float64, n*k)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	b := tensor.FromFloat64(tensor.Shape{n, k}, bd)
	x, err := linalg.Solve(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !x.Shape().Equal(tensor.Shape{n, k}) {
		t.Fatalf("X shape %v", x.Shape())
	}
	prod := matmul(a, x)
	for i := range n {
		for j := range k {
			if math.Abs(prod.AtF64(i, j)-b.AtF64(i, j)) > 1e-9 {
				t.Errorf("(A·X)[%d,%d] = %g, want %g", i, j, prod.AtF64(i, j), b.AtF64(i, j))
			}
		}
	}
}

// §V16 tier-1: a singular matrix factorizes (Det == 0) but Solve/Inverse error; non-square/empty error.
func TestSingularAndShapeErrors(t *testing.T) {
	sing := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 4}) // row2 = 2·row1
	f, err := linalg.Factor(sing)
	if err != nil {
		t.Fatalf("Factor of singular should succeed: %v", err)
	}
	if d := f.Det(); math.Abs(d) > 1e-12 {
		t.Errorf("det(singular) = %g, want 0", d)
	}
	if _, err := f.Inverse(); err == nil {
		t.Error("Inverse of singular matrix must error")
	}
	if _, err := linalg.Solve(sing, tensor.FromFloat64(tensor.Shape{2}, []float64{1, 1})); err == nil {
		t.Error("Solve of singular matrix must error")
	}
	if _, err := linalg.Factor(tensor.New(tensor.F64, tensor.Shape{2, 3})); err == nil {
		t.Error("non-square matrix must error")
	}
}

// ExampleLU factorizes once and reuses the factorization for both the determinant and a solve.
func ExampleLU() {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 1, 1, 3})
	f, _ := linalg.Factor(a)
	x, _ := f.Solve(tensor.FromFloat64(tensor.Shape{2}, []float64{3, 5}))
	fmt.Printf("det=%.0f x=[%.1f %.1f]\n", f.Det(), x.AtF64(0), x.AtF64(1))
	// Output: det=5 x=[0.8 1.4]
}

func ExampleSolve() {
	// solve [[2,1],[1,3]]·x = [3,5] → x = [0.8, 1.4]
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 1, 1, 3})
	b := tensor.FromFloat64(tensor.Shape{2}, []float64{3, 5})
	x, _ := linalg.Solve(a, b)
	fmt.Printf("%.1f %.1f\n", x.AtF64(0), x.AtF64(1))
	// Output: 0.8 1.4
}

func ExampleDet() {
	d, _ := linalg.Det(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4}))
	fmt.Printf("%.0f\n", d)
	// Output: -2
}

func ExampleInverse() {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{4, 7, 2, 6})
	inv, _ := linalg.Inverse(a) // 1/(4·6−7·2)·[[6,−7],[−2,4]] = 0.1·[[6,−7],[−2,4]]
	fmt.Printf("%.1f %.1f %.1f %.1f\n", inv.AtF64(0, 0), inv.AtF64(0, 1), inv.AtF64(1, 0), inv.AtF64(1, 1))
	// Output: 0.6 -0.7 -0.2 0.4
}
