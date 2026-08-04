package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// solveSPDKernel solves the symmetric positive-definite system A·X = B for X via
// the Cholesky factor A = L·Lᵀ (LAPACK dpotrs / numpy.linalg.solve on SPD input):
// forward-substitute L·Y = B, then back-substitute Lᵀ·X = Y. B (and X) is a vector
// [n] or a matrix [n,k] of right-hand sides. All arithmetic is f64 (§V10); the SPD
// and shape errors on A come from the shared Cholesky factorization.
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func solveSPDKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: solvespd wants 2 inputs (A, B), got %d", len(in))
	}
	a, b := in[0], in[1]
	lout, err := choleskyKernel(ctx, in[:1], nil) // validates square + SPD
	if err != nil {
		return nil, err
	}
	l := lout[0]
	n := l.Shape()[0]

	// right-hand side geometry: vector [n] (k=1) or matrix [n,k].
	vector := b.Ndim() == 1
	var k int
	switch {
	case vector && b.Shape()[0] == n:
		k = 1
	case b.Ndim() == 2 && b.Shape()[0] == n:
		k = b.Shape()[1]
	default:
		return nil, fmt.Errorf("ref: solvespd rhs must be [%d] or [%d,k], got %v", n, n, b.Shape())
	}
	rhs := func(i, c int) float64 {
		if vector {
			return b.AtF64(i)
		}
		return b.AtF64(i, c)
	}

	// y = L⁻¹·B (forward substitution), then x = L⁻ᵀ·y (back substitution), per column.
	// Flat [n*k] row-major buffers instead of n allocated rows: both substitutions
	// index y[p][c] and x[p][c] with p varying — a column walk that, in a
	// row-of-slices layout, dereferences a different heap row per step. Reading L
	// through a flat typed view additionally removes an AtF64 dispatch from the
	// O(n²k) innermost loop, which is where the dominant cost was. Index arithmetic
	// and identical operand order throughout, so results are bit-identical; the
	// accessor form is kept for dtypes f64Data cannot expose.
	x := make([]float64, n*k)
	y := make([]float64, n*k)
	lf, lok := f64Data(l)
	lat := func(i, j int) float64 {
		if lok {
			return lf[i*n+j]
		}
		return l.AtF64(i, j)
	}
	// The right-hand-side column is the INNER loop, not the outer one. With c outermost every step
	// of the substitution jumped k elements through y and x and re-fetched one L element per
	// innermost iteration; with c innermost each step walks two contiguous rows and the L element
	// is loop-invariant, loaded once per (i,p) instead of once per (i,p,c).
	//
	// BIT-IDENTICAL: for each output (i,c) the subtractions still run over ascending p with the
	// same two operands and the final divide uses the same L[i,i]. Only the interleaving of
	// INDEPENDENT (i,c) results changes, and the accumulator moving from a register to a float64
	// slot is exact on this target — there are no extended-precision intermediates.
	for i := range n {
		yi := y[i*k : i*k+k]
		for c := range k {
			yi[c] = rhs(i, c)
		}
		for p := range i {
			lip, yp := lat(i, p), y[p*k:p*k+k]
			for c := range k {
				//perfscan:ignore PS3075 reference oracle: intentionally simple, correctness baseline not an optimization target
				yi[c] -= lip * yp[c]
			}
		}
		d := lat(i, i)
		//perfscan:ignore PS5001 reference oracle: intentionally simple, correctness baseline not an optimization target
		for c := range k {
			yi[c] /= d
		}
	}
	for i := n - 1; i >= 0; i-- {
		xi := x[i*k : i*k+k]
		copy(xi, y[i*k:i*k+k])
		for p := i + 1; p < n; p++ {
			lpi, xp := lat(p, i), x[p*k:p*k+k] // Lᵀ[i,p] = L[p,i]
			for c := range k {
				//perfscan:ignore PS3075 reference oracle: intentionally simple, correctness baseline not an optimization target
				xi[c] -= lpi * xp[c]
			}
		}
		d := lat(i, i)
		//perfscan:ignore PS5001 reference oracle: intentionally simple, correctness baseline not an optimization target
		for c := range k {
			xi[c] /= d
		}
	}

	out := tensor.NewOn(ctx.Device(), a.Dtype(), b.Shape())
	for i := range n {
		for c := range k {
			if vector {
				out.SetF64(x[i*k+c], i)
			} else {
				out.SetF64(x[i*k+c], i, c)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpSolveSPD, tensor.F32, solveSPDKernel)
	std.add(backend.OpSolveSPD, tensor.F64, solveSPDKernel)
}
