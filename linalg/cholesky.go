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
	l, err := cholFactor(a, n)
	if err != nil {
		return nil, err
	}
	out := make([]float64, n*n)
	for i := range n {
		for j := 0; j <= i; j++ {
			out[i*n+j] = l[i][j]
		}
	}
	return tensor.FromFloat64(tensor.Shape{n, n}, out), nil
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
	l, err := cholFactor(a, n)
	if err != nil {
		return nil, err
	}
	vec := b.Ndim() == 1
	cols := 1
	if !vec {
		cols = b.Shape()[1]
	}
	out := make([]float64, n*cols) // [n,cols] row-major
	// Allocated ONCE for all columns rather than per column, matching what LU.Solve
	// already does. Safe because the buffer is fully overwritten at the start of its own
	// pass before anything reads it, so it cannot leak the previous column's values. It is
	// a function local, not a receiver field: that keeps concurrent calls independent
	// (PS6006 — a receiver slice used as per-call scratch is a data race waiting for its
	// second caller). Measured at n=512, cols=512: 1032 allocations per call down to 521,
	// B/op 8.40MB down to 6.31MB.
	y := make([]float64, n)
	for c := range cols {
		for i := range n { // forward: L·y = b
			var s float64
			if vec {
				s = b.AtF64(i)
			} else {
				s = b.AtF64(i, c)
			}
			for k := range i {
				s -= l[i][k] * y[k]
			}
			y[i] = s / l[i][i]
		}
		for i := n - 1; i >= 0; i-- { // back: Lᵀ·x = y (Lᵀ[i,k] = L[k,i])
			s := y[i]
			for k := i + 1; k < n; k++ {
				s -= l[k][i] * out[k*cols+c]
			}
			out[i*cols+c] = s / l[i][i]
		}
	}
	if vec {
		return tensor.FromFloat64(tensor.Shape{n}, out), nil
	}
	return tensor.FromFloat64(tensor.Shape{n, cols}, out), nil
}

// cholFactor computes the lower-triangular Cholesky factor L (as [][]float64) of the n×n SPD matrix a,
// erroring if a is not positive definite.
func cholFactor(a *tensor.Tensor, n int) ([][]float64, error) {
	l := make([][]float64, n)
	for i := range n {
		l[i] = make([]float64, n)
	}
	for j := range n {
		d := a.AtF64(j, j)
		for k := range j {
			d -= l[j][k] * l[j][k]
		}
		if d <= 0 {
			return nil, fmt.Errorf("linalg: matrix is not positive definite (non-positive pivot %g at leading minor %d)", d, j)
		}
		l[j][j] = math.Sqrt(d)
		for i := j + 1; i < n; i++ {
			s := a.AtF64(i, j)
			for k := range j {
				s -= l[i][k] * l[j][k]
			}
			l[i][j] = s / l[j][j]
		}
	}
	return l, nil
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
