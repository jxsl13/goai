package linalg_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

func transpose(a *tensor.Tensor) *tensor.Tensor {
	m, n := a.Shape()[0], a.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{n, m})
	for i := range m {
		for j := range n {
			out.SetF64(a.AtF64(i, j), j, i)
		}
	}
	return out
}

func approxEqual(t *testing.T, name string, x, y *tensor.Tensor, tol float64) {
	t.Helper()
	if !x.Shape().Equal(y.Shape()) {
		t.Fatalf("%s: shape %v != %v", name, x.Shape(), y.Shape())
	}
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		if math.Abs(x.AtF64(idx...)-y.AtF64(idx...)) > tol {
			t.Errorf("%s[%v] = %g vs %g", name, idx, x.AtF64(idx...), y.AtF64(idx...))
		}
	}
}

// §V16 tier-1: the pseudoinverse satisfies the four Moore-Penrose conditions (§R118) across tall,
// square and wide matrices, and for full-rank tall A it equals the least-squares left-inverse.
func TestPinvMoorePenrose(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 62))
	for _, d := range [][2]int{{3, 3}, {6, 4}, {4, 6}, {8, 2}, {5, 5}} {
		m, n := d[0], d[1]
		ad := make([]float64, m*n)
		for i := range ad {
			ad[i] = rng.NormFloat64()
		}
		a := tensor.FromFloat64(tensor.Shape{m, n}, ad)
		ap, err := linalg.Pinv(a)
		if err != nil {
			t.Fatalf("%dx%d: %v", m, n, err)
		}
		if !ap.Shape().Equal(tensor.Shape{n, m}) {
			t.Fatalf("%dx%d A⁺ shape %v", m, n, ap.Shape())
		}
		aap := matmul(a, ap) // A·A⁺  (m×m)
		apa := matmul(ap, a) // A⁺·A  (n×n)
		approxEqual(t, "A·A⁺·A=A", matmul(aap, a), a, 1e-8)
		approxEqual(t, "A⁺·A·A⁺=A⁺", matmul(apa, ap), ap, 1e-8)
		approxEqual(t, "(A·A⁺)ᵀ=A·A⁺", transpose(aap), aap, 1e-8)
		approxEqual(t, "(A⁺·A)ᵀ=A⁺·A", transpose(apa), apa, 1e-8)
	}
	// full-rank tall: A⁺·b == least-squares solution
	rng2 := rand.New(rand.NewPCG(9, 9))
	m, n := 10, 3
	ad := make([]float64, m*n)
	bd := make([]float64, m)
	for i := range ad {
		ad[i] = rng2.NormFloat64()
	}
	for i := range bd {
		bd[i] = rng2.NormFloat64()
	}
	a := tensor.FromFloat64(tensor.Shape{m, n}, ad)
	b := tensor.FromFloat64(tensor.Shape{m}, bd)
	ap, _ := linalg.Pinv(a)
	// A⁺·b
	pinvB := make([]float64, n)
	for i := range n {
		for j := range m {
			pinvB[i] += ap.AtF64(i, j) * b.AtF64(j)
		}
	}
	ls, _ := linalg.Lstsq(a, b)
	for i := range n {
		if math.Abs(pinvB[i]-ls.AtF64(i)) > 1e-8 {
			t.Errorf("A⁺·b[%d]=%g != Lstsq %g", i, pinvB[i], ls.AtF64(i))
		}
	}
}

// §V16 tier-1: for a square invertible matrix, Pinv equals the inverse.
func TestPinvEqualsInverse(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 7))
	a := randMatrix(rng, 5)
	ap, _ := linalg.Pinv(a)
	inv, _ := linalg.Inverse(a)
	approxEqual(t, "Pinv=Inverse", ap, inv, 1e-8)
}

// §V16 tier-1: numerical rank on known matrices (§R118).
func TestRank(t *testing.T) {
	if r, _ := linalg.Rank(tensor.Eye(tensor.F64, 4)); r != 4 {
		t.Errorf("rank(I₄) = %d, want 4", r)
	}
	// rank-1: outer product [1,2,3]ᵀ·[1,1,1]
	r1 := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{1, 1, 1, 2, 2, 2, 3, 3, 3})
	if r, _ := linalg.Rank(r1); r != 1 {
		t.Errorf("rank(rank-1) = %d, want 1", r)
	}
	if r, _ := linalg.Rank(tensor.Zeros(tensor.F64, tensor.Shape{3, 3})); r != 0 {
		t.Errorf("rank(0) = %d, want 0", r)
	}
	// singular 3×3 (row3 = row1+row2) → rank 2
	sing := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{1, 0, 2, 0, 1, 3, 1, 1, 5})
	if r, _ := linalg.Rank(sing); r != 2 {
		t.Errorf("rank(singular) = %d, want 2", r)
	}
}

// §V16 tier-1: condition number and matrix norms on known cases (§R118).
func TestCondAndNorms(t *testing.T) {
	if c, _ := linalg.Cond(tensor.Eye(tensor.F64, 3)); math.Abs(c-1) > 1e-9 {
		t.Errorf("cond(I) = %g, want 1", c)
	}
	// diagonal(2,-5,3): σ = 5,3,2 → cond = 5/2 = 2.5, ‖‖₂=5
	diag := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{2, 0, 0, 0, -5, 0, 0, 0, 3})
	if c, _ := linalg.Cond(diag); math.Abs(c-2.5) > 1e-9 {
		t.Errorf("cond(diag) = %g, want 2.5", c)
	}
	if s, _ := linalg.Norm2(diag); math.Abs(s-5) > 1e-9 {
		t.Errorf("‖diag‖₂ = %g, want 5", s)
	}
	// singular → +Inf
	sing := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 4})
	if c, _ := linalg.Cond(sing); !math.IsInf(c, 1) {
		t.Errorf("cond(singular) = %g, want +Inf", c)
	}
	// A = [[3,-4],[0,0]]: Fro=5, ‖‖₁=max col=|3|+0=3 vs |−4|+0=4 → 4, ‖‖∞=max row=|3|+|−4|=7 vs 0 → 7
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{3, -4, 0, 0})
	if f, _ := linalg.NormFro(a); math.Abs(f-5) > 1e-12 {
		t.Errorf("‖A‖_F = %g, want 5", f)
	}
	if n1, _ := linalg.Norm1(a); math.Abs(n1-4) > 1e-12 {
		t.Errorf("‖A‖₁ = %g, want 4", n1)
	}
	if ni, _ := linalg.NormInf(a); math.Abs(ni-7) > 1e-12 {
		t.Errorf("‖A‖_∞ = %g, want 7", ni)
	}
}

func ExamplePinv() {
	// pseudoinverse of a 2×1 column [3;4]: A⁺ = [3,4]/25 = [0.12, 0.16]
	a := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{3, 4})
	ap, _ := linalg.Pinv(a)
	fmt.Printf("%.2f %.2f\n", ap.AtF64(0, 0), ap.AtF64(0, 1))
	// Output: 0.12 0.16
}

func ExampleNormFro() {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{3, 0, 0, 4})
	f, _ := linalg.NormFro(a)
	fmt.Printf("%.0f\n", f)
	// Output: 5
}

func ExampleRank() {
	// rank-1 matrix
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 4})
	r, _ := linalg.Rank(a)
	fmt.Println(r)
	// Output: 1
}
