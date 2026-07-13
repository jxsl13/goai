// Package linalg provides gonum/numpy-style dense linear algebra over rank-2 tensors: an LU
// factorization with partial pivoting (P·A = L·U) and the derived determinant, linear solve, and
// matrix inverse. The algorithms are the standard textbook ones (Golub & Van Loan, "Matrix
// Computations", §3.2 LU/GEPP and §3.4 determinant; LAPACK dgetrf/dgetrs/dgetri), §R115. Everything
// is computed in f64; inputs are read via AtF64 and results are fresh f64 tensors.
package linalg

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// LU is a partial-pivoting LU factorization P·A = L·U of a square matrix: L is unit-lower-triangular
// (implicit 1s on the diagonal), U is upper-triangular, and P is a row permutation. L and U are
// packed into a single n×n array (L below the diagonal, U on and above). Reuse it to solve several
// right-hand sides or to get the determinant without refactoring.
type LU struct {
	lu   [][]float64 // packed L (strictly-lower, unit) + U (upper), n×n
	piv  []int       // row permutation: row i of P·A is original row piv[i]
	sign float64     // (−1)^(#row swaps), for the determinant
	n    int
}

// Factor computes the partial-pivoting LU factorization of the square matrix a (right-looking
// Doolittle: at each column pick the max-|·| pivot, swap, then eliminate). It succeeds even for a
// singular matrix — the singularity then surfaces as Det()==0 and an error from Solve/Inverse.
func Factor(a *tensor.Tensor) (*LU, error) {
	n, err := squareN(a)
	if err != nil {
		return nil, err
	}
	m := toMatrix(a, n)
	piv := make([]int, n)
	for i := range piv {
		piv[i] = i
	}
	sign := 1.0
	for k := range n {
		// partial pivot: largest absolute value in column k, rows k..n−1 (numerical stability)
		p := k
		for i := k + 1; i < n; i++ {
			if math.Abs(m[i][k]) > math.Abs(m[p][k]) {
				p = i
			}
		}
		if m[p][k] == 0 {
			continue // singular column: U[k,k] stays 0, no elimination
		}
		if p != k {
			m[k], m[p] = m[p], m[k]
			piv[k], piv[p] = piv[p], piv[k]
			sign = -sign
		}
		for i := k + 1; i < n; i++ {
			mult := m[i][k] / m[k][k]
			m[i][k] = mult // store the L multiplier
			for j := k + 1; j < n; j++ {
				m[i][j] -= mult * m[k][j]
			}
		}
	}
	return &LU{lu: m, piv: piv, sign: sign, n: n}, nil
}

// Det returns the determinant det(A) = sign · Π_k U[k,k] (the permutation sign times the product of
// the U diagonal). It is 0 exactly when A is singular.
func (f *LU) Det() float64 {
	d := f.sign
	for k := range f.n {
		d *= f.lu[k][k]
	}
	return d
}

// Solve solves A·X = B for X, where B is a right-hand-side vector [n] or matrix [n,k]. It applies the
// row permutation to B, then forward-substitution (L·Y = P·B) and back-substitution (U·X = Y).
// Returns an error if A is singular (a zero on the U diagonal).
func (f *LU) Solve(b *tensor.Tensor) (*tensor.Tensor, error) {
	n := f.n
	if b.Ndim() < 1 || b.Ndim() > 2 || b.Shape()[0] != n {
		return nil, fmt.Errorf("linalg: Solve rhs must be [%d] or [%d,k], got %v", n, n, b.Shape())
	}
	for k := range n {
		if f.lu[k][k] == 0 {
			return nil, fmt.Errorf("linalg: Solve of a singular matrix (zero pivot at %d)", k)
		}
	}
	vec := b.Ndim() == 1
	cols := 1
	if !vec {
		cols = b.Shape()[1]
	}
	bat := func(i, c int) float64 {
		if vec {
			return b.AtF64(i)
		}
		return b.AtF64(i, c)
	}
	out := make([]float64, n*cols) // row-major [n,cols]
	for c := range cols {
		y := make([]float64, n)
		for i := range n { // forward: L·y = P·b, L unit-lower
			s := bat(f.piv[i], c)
			for j := range i {
				s -= f.lu[i][j] * y[j]
			}
			y[i] = s
		}
		for i := n - 1; i >= 0; i-- { // back: U·x = y
			s := y[i]
			for j := i + 1; j < n; j++ {
				s -= f.lu[i][j] * out[j*cols+c]
			}
			out[i*cols+c] = s / f.lu[i][i]
		}
	}
	if vec {
		return tensor.FromFloat64(tensor.Shape{n}, out), nil
	}
	return tensor.FromFloat64(tensor.Shape{n, cols}, out), nil
}

// Inverse returns A⁻¹ by solving A·X = I with the existing factorization (one LU + n substitutions).
// Errors if A is singular. (Prefer Solve when you only need A⁻¹·b; the explicit inverse is for when
// the inverse itself is wanted.)
func (f *LU) Inverse() (*tensor.Tensor, error) {
	return f.Solve(tensor.Eye(tensor.F64, f.n))
}

// Solve solves A·X = B in one shot (Factor then Solve). B is a vector [n] or matrix [n,k].
func Solve(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	f, err := Factor(a)
	if err != nil {
		return nil, err
	}
	return f.Solve(b)
}

// Inverse returns the inverse of the square matrix a (Factor then Inverse).
func Inverse(a *tensor.Tensor) (*tensor.Tensor, error) {
	f, err := Factor(a)
	if err != nil {
		return nil, err
	}
	return f.Inverse()
}

// Det returns the determinant of the square matrix a (Factor then Det).
func Det(a *tensor.Tensor) (float64, error) {
	f, err := Factor(a)
	if err != nil {
		return 0, err
	}
	return f.Det(), nil
}

// squareN validates that a is a rank-2 square matrix and returns its size.
func squareN(a *tensor.Tensor) (int, error) {
	if a.Ndim() != 2 || a.Shape()[0] != a.Shape()[1] {
		return 0, fmt.Errorf("linalg: expected a square rank-2 matrix, got %v", a.Shape())
	}
	if a.Shape()[0] == 0 {
		return 0, fmt.Errorf("linalg: empty matrix")
	}
	return a.Shape()[0], nil
}

// toMatrix copies a into a fresh n×n [][]float64 (so the factorization never mutates the input).
func toMatrix(a *tensor.Tensor, n int) [][]float64 {
	m := make([][]float64, n)
	for i := range n {
		m[i] = make([]float64, n)
		for j := range n {
			m[i][j] = a.AtF64(i, j)
		}
	}
	return m
}
