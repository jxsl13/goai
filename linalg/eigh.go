package linalg

import (
	"fmt"
	"math"

	intlinalg "github.com/jxsl13/goai/internal/linalg"
	"github.com/jxsl13/goai/tensor"
)

// Eigh computes the eigendecomposition of a real SYMMETRIC matrix, A = V·diag(w)·Vᵀ
// (numpy.linalg.eigh / LAPACK dsyev, §R119): w holds the real eigenvalues in ASCENDING order
// (w₀ ≤ w₁ ≤ … ≤ wₙ₋₁) and the COLUMNS of V are the corresponding orthonormal eigenvectors, so
// A·V[:,k] = w[k]·V[:,k] and VᵀV = I. It uses the cyclic (two-sided) Jacobi eigenvalue method, which
// is unconditionally stable for symmetric input and computes small eigenvalues to high relative
// accuracy. a must be square and symmetric (checked to a relative tolerance). Returns fresh f64
// tensors: w of shape [n] and V of shape [n,n]; the input is not mutated.
func Eigh(a *tensor.Tensor) (w, v *tensor.Tensor, err error) {
	n, err := squareN(a)
	if err != nil {
		return nil, nil, err
	}
	ma := toMatrix(a, n)
	for i := range n {
		for j := i + 1; j < n; j++ {
			if math.Abs(ma[i][j]-ma[j][i]) > 1e-9*(math.Abs(ma[i][j])+math.Abs(ma[j][i]))+1e-12 {
				return nil, nil, fmt.Errorf("linalg: Eigh needs a symmetric matrix; A[%d,%d]=%g != A[%d,%d]=%g",
					i, j, ma[i][j], j, i, ma[j][i])
			}
		}
	}
	vals, vecs := intlinalg.SymEig(ma) // eigenvalues DESCENDING; vecs[k] = k-th eigenvector
	wOut := make([]float64, n)
	vOut := make([]float64, n*n)
	for k := range n {
		src := n - 1 - k // ascending index k ← descending index n−1−k
		wOut[k] = vals[src]
		for r := range n {
			vOut[r*n+k] = vecs[src][r] // column k of V is the k-th (ascending) eigenvector
		}
	}
	return tensor.FromFloat64(tensor.Shape{n}, wOut), tensor.FromFloat64(tensor.Shape{n, n}, vOut), nil
}
