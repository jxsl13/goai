package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// choleskyKernel computes the lower-triangular Cholesky factor L of a symmetric
// positive-definite matrix A, with A = L·Lᵀ (numpy.linalg.cholesky, uplo='L').
// Only the lower triangle of A is read (the upper is assumed symmetric, matching
// LAPACK potrf); the returned L is lower-triangular with a positive diagonal and
// exact zeros above it. The Cholesky–Banachiewicz recurrence accumulates in
// float64 (§V10); an F32 input is computed in f64 and narrowed on store. A
// non-positive pivot means A is not positive-definite and is reported as an error.
func choleskyKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: cholesky wants 1 input, got %d", len(in))
	}
	a := in[0]
	if a.Ndim() != 2 || a.Shape()[0] != a.Shape()[1] {
		return nil, fmt.Errorf("ref: cholesky needs a square matrix, got shape %v", a.Shape())
	}
	n := a.Shape()[0]
	l := make([][]float64, n) // lower factor, dense f64 working copy
	for i := range n {
		l[i] = make([]float64, n)
	}
	for j := range n {
		// diagonal: L[j,j] = √(A[j,j] − Σ_{k<j} L[j,k]²)
		d := a.AtF64(j, j)
		for k := range j {
			d -= l[j][k] * l[j][k]
		}
		if d <= 0 {
			return nil, fmt.Errorf("ref: cholesky: matrix is not positive-definite (non-positive pivot %.6g at %d)", d, j)
		}
		ljj := math.Sqrt(d)
		l[j][j] = ljj
		// below the diagonal: L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]
		for i := j + 1; i < n; i++ {
			s := a.AtF64(i, j)
			for k := range j {
				s -= l[i][k] * l[j][k]
			}
			l[i][j] = s / ljj
		}
	}
	out := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n, n})
	for i := range n {
		for j := 0; j <= i; j++ {
			out.SetF64(l[i][j], i, j)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpCholesky, tensor.F32, choleskyKernel)
	std.add(backend.OpCholesky, tensor.F64, choleskyKernel)
}
