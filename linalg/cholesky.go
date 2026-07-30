package linalg

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Cholesky returns the lower-triangular Cholesky factor L of a real symmetric positive-definite (SPD)
// matrix, A = L·Lᵀ with L[i,i] > 0 (numpy.linalg.cholesky / LAPACK dpotrf, §R120). It uses the
// Cholesky-Banachiewicz recurrence (no pivoting — SPD needs none) and returns an error exactly when A
// is not positive definite (a non-positive √ argument at some leading minor). a must be square and
// symmetric; it is not mutated. Cholesky is ~2× faster than LU and is the standard direct solver for
// SPD systems (normal equations, covariance, Gaussian processes).
func Cholesky(a *tensor.Tensor) (*tensor.Tensor, error) {
	n, err := squareN(a)
	if err != nil {
		return nil, err
	}
	if err := requireSymmetric(a, n); err != nil {
		return nil, err
	}
	lf, err := cholFactor(a, n)
	if err != nil {
		return nil, err
	}
	// lf is already the lower-triangular L in row-major order: cholFactor writes
	// only the diagonal and sub-diagonal, leaving the strictly-upper entries at
	// their make() zero, so it IS the returned factor (no lower-triangle copy).
	return tensor.FromFloat64(tensor.Shape{n, n}, lf), nil
}

// CholSolve solves the SPD system A·X = B via the Cholesky factorization (LAPACK dpotrs): forward-
// substitution L·Y = B, then back-substitution Lᵀ·X = Y. B is a vector [n] or matrix [n,k]; errors if
// A is not SPD.
func CholSolve(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	n, err := squareN(a)
	if err != nil {
		return nil, err
	}
	if b.Ndim() < 1 || b.Ndim() > 2 || b.Shape()[0] != n {
		return nil, fmt.Errorf("linalg: CholSolve rhs must be [%d] or [%d,k], got %v", n, n, b.Shape())
	}
	if err := requireSymmetric(a, n); err != nil {
		return nil, err
	}
	lf, err := cholFactor(a, n)
	if err != nil {
		return nil, err
	}
	vec := b.Ndim() == 1
	cols := 1
	if !vec {
		cols = b.Shape()[1]
	}
	out := make([]float64, n*cols) // [n,cols] row-major
	for c := range cols {
		y := make([]float64, n)
		for i := range n { // forward: L·y = b
			var s float64
			if vec {
				s = b.AtF64(i)
			} else {
				s = b.AtF64(i, c)
			}
			li := lf[i*n : i*n+n] // contiguous factor row i (was l[i], a scattered slice)
			for k := range i {
				s -= li[k] * y[k]
			}
			y[i] = s / li[i]
		}
		for i := n - 1; i >= 0; i-- { // back: Lᵀ·x = y (Lᵀ[i,k] = L[k,i])
			s := y[i]
			for k := i + 1; k < n; k++ {
				s -= lf[k*n+i] * out[k*cols+c] // L[k,i], a strided column read (unchanged)
			}
			out[i*cols+c] = s / lf[i*n+i]
		}
	}
	if vec {
		return tensor.FromFloat64(tensor.Shape{n}, out), nil
	}
	return tensor.FromFloat64(tensor.Shape{n, cols}, out), nil
}

// cholFactor computes the lower-triangular Cholesky factor L of the n×n SPD matrix a as a FLAT
// row-major []float64 (L[i][j] at lf[i*n+j]), erroring if a is not positive definite. The flat
// buffer replaces the previous [][]float64 (one heap slice per row, scattered in memory with a
// slice-header dereference on every l[i][k] access): a single contiguous block gives the O(n³)
// inner dot-product sequential, pointer-chase-free reads of the already-computed columns. Arithmetic
// is byte-for-byte the Cholesky-Banachiewicz recurrence: same ascending-k summation order and the
// same `s / L[j][j]` DIVISION (not a hoisted reciprocal, which would reround), so L is bit-identical.
func cholFactor(a *tensor.Tensor, n int) ([]float64, error) {
	lf := make([]float64, n*n)
	for j := range n {
		lj := lf[j*n : j*n+n] // contiguous row j
		d := a.AtF64(j, j)
		for k := range j {
			d -= lj[k] * lj[k]
		}
		if d <= 0 {
			return nil, fmt.Errorf("linalg: matrix is not positive definite (non-positive pivot %g at leading minor %d)", d, j)
		}
		ljj := math.Sqrt(d)
		lj[j] = ljj
		for i := j + 1; i < n; i++ {
			li := lf[i*n : i*n+n] // contiguous row i
			s := a.AtF64(i, j)
			for k := range j {
				s -= li[k] * lj[k]
			}
			li[j] = s / ljj
		}
	}
	return lf, nil
}

// requireSymmetric errors if a is not symmetric to a relative tolerance.
func requireSymmetric(a *tensor.Tensor, n int) error {
	for i := range n {
		for j := i + 1; j < n; j++ {
			if math.Abs(a.AtF64(i, j)-a.AtF64(j, i)) > 1e-9*(math.Abs(a.AtF64(i, j))+math.Abs(a.AtF64(j, i)))+1e-12 {
				return fmt.Errorf("linalg: expected a symmetric matrix; A[%d,%d]=%g != A[%d,%d]=%g",
					i, j, a.AtF64(i, j), j, i, a.AtF64(j, i))
			}
		}
	}
	return nil
}
