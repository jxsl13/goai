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
	x := make([][]float64, n)
	y := make([][]float64, n)
	for i := range n {
		x[i] = make([]float64, k)
		y[i] = make([]float64, k)
	}
	for c := range k {
		for i := range n {
			s := rhs(i, c)
			for p := range i {
				s -= l.AtF64(i, p) * y[p][c]
			}
			y[i][c] = s / l.AtF64(i, i)
		}
		for i := n - 1; i >= 0; i-- {
			s := y[i][c]
			for p := i + 1; p < n; p++ {
				s -= l.AtF64(p, i) * x[p][c] // Lᵀ[i,p] = L[p,i]
			}
			x[i][c] = s / l.AtF64(i, i)
		}
	}

	out := tensor.NewOn(ctx.Device(), a.Dtype(), b.Shape())
	for i := range n {
		for c := range k {
			if vector {
				out.SetF64(x[i][c], i)
			} else {
				out.SetF64(x[i][c], i, c)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpSolveSPD, tensor.F32, solveSPDKernel)
	std.add(backend.OpSolveSPD, tensor.F64, solveSPDKernel)
}
