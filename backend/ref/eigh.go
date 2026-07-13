package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	intlinalg "github.com/jxsl13/goai/internal/linalg"
	"github.com/jxsl13/goai/tensor"
)

// eighKernel computes the eigendecomposition of a real SYMMETRIC matrix,
// A = V·diag(w)·Vᵀ (numpy.linalg.eigh / LAPACK dsyev): w holds the real eigenvalues
// ASCENDING and the COLUMNS of V are the orthonormal eigenvectors (VᵀV = I,
// A·V[:,k] = w[k]·V[:,k]). It symmetrizes the input to (A+Aᵀ)/2 and runs the cyclic
// Jacobi method (internal/linalg.SymEig — shared kernel, also used by PCA/GaLore),
// then reverses SymEig's descending order to ascending. Two outputs [w, V]; f64
// arithmetic (§V10), F32 input factors in f64 and narrows on store.
func eighKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: eigh wants 1 input, got %d", len(in))
	}
	a := in[0]
	if a.Ndim() != 2 || a.Shape()[0] != a.Shape()[1] {
		return nil, fmt.Errorf("ref: eigh needs a square matrix, got shape %v", a.Shape())
	}
	n := a.Shape()[0]
	ma := make([][]float64, n) // symmetric part (A+Aᵀ)/2 → robust to tiny asymmetry
	for i := range n {
		ma[i] = make([]float64, n)
	}
	for i := range n {
		for j := range n {
			ma[i][j] = 0.5 * (a.AtF64(i, j) + a.AtF64(j, i))
		}
	}
	vals, vecs := intlinalg.SymEig(ma) // eigenvalues DESCENDING; vecs[k] = k-th eigenvector

	wt := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n})
	vt := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n, n})
	for k := range n {
		src := n - 1 - k // ascending index k ← descending index n−1−k
		wt.SetF64(vals[src], k)
		for r := range n {
			vt.SetF64(vecs[src][r], r, k) // column k of V is the k-th (ascending) eigenvector
		}
	}
	return []*tensor.Tensor{wt, vt}, nil
}

func init() {
	std.add(backend.OpEigh, tensor.F32, eighKernel)
	std.add(backend.OpEigh, tensor.F64, eighKernel)
}
