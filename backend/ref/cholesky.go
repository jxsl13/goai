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
	// The lower factor is ONE flat [n*n] row-major buffer, not a slice of rows. The
	// O(n³) inner product below walks two rows of it per step; with [][]float64 each
	// step chases a separate pointer to a separately allocated row, which costs both
	// the indirection and any cache locality between rows. Flattening also drops n
	// allocations to 1. No arithmetic changes — same values, same order, same
	// accumulation — so the factor is bit-identical.
	l := make([]float64, n*n)
	// Read the input through a flat typed view where possible (exact widening for
	// F32); the per-element AtF64 dispatch is retained as the fallback.
	as, aok := f64Data(a)
	at := func(i, j int) float64 {
		if aok {
			return as[i*n+j]
		}
		return a.AtF64(i, j)
	}
	for j := range n {
		lj := l[j*n : j*n+n]
		// diagonal: L[j,j] = √(A[j,j] − Σ_{k<j} L[j,k]²)
		d := at(j, j)
		for k := range j {
			d -= lj[k] * lj[k]
		}
		if d <= 0 {
			return nil, fmt.Errorf("ref: cholesky: matrix is not positive-definite (non-positive pivot %.6g at %d)", d, j)
		}
		ljj := math.Sqrt(d)
		lj[j] = ljj
		// below the diagonal: L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]
		for i := j + 1; i < n; i++ {
			li := l[i*n : i*n+n]
			s := at(i, j)
			for k := range j {
				s -= li[k] * lj[k]
			}
			li[j] = s / ljj
		}
	}
	out := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n, n})
	if os, flush, ok := outF64(out); ok {
		for i := range n {
			for j := 0; j <= i; j++ {
				os[i*n+j] = l[i*n+j]
			}
		}
		flush()
	} else {
		for i := range n {
			for j := 0; j <= i; j++ {
				out.SetF64(l[i*n+j], i, j)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpCholesky, tensor.F32, choleskyKernel)
	std.add(backend.OpCholesky, tensor.F64, choleskyKernel)
}
