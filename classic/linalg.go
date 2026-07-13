// Package classic is layer L4: classical machine learning — OLS linear
// regression, softmax (multinomial logistic) regression, k-means (Lloyd), and
// PCA (Jacobi eigendecomposition). All f64; goldens come from real sklearn
// (§T25, §V1).
package classic

import (
	"fmt"
	"math"
)

// cholSolve solves the symmetric positive-definite system A·x = b in place via
// Cholesky decomposition (A = L·Lᵀ) — the standard direct solver for
// normal-equation OLS (Golub & Van Loan, Matrix Computations §4.2).
func cholSolve(a [][]float64, b []float64) ([]float64, error) {
	n := len(a)
	l := make([][]float64, n)
	for i := range l {
		l[i] = make([]float64, n)
	}
	for i := range n {
		for j := 0; j <= i; j++ {
			sum := a[i][j]
			for k := range j {
				sum -= l[i][k] * l[j][k]
			}
			if i == j {
				if sum <= 0 {
					return nil, fmt.Errorf("classic: matrix not positive definite (pivot %g at %d)", sum, i)
				}
				l[i][i] = math.Sqrt(sum)
			} else {
				l[i][j] = sum / l[j][j]
			}
		}
	}
	// forward substitution L·z = b
	z := make([]float64, n)
	for i := range n {
		s := b[i]
		for k := range i {
			s -= l[i][k] * z[k]
		}
		z[i] = s / l[i][i]
	}
	// back substitution Lᵀ·x = z
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := z[i]
		for k := i + 1; k < n; k++ {
			s -= l[k][i] * x[k]
		}
		x[i] = s / l[i][i]
	}
	return x, nil
}

// The symmetric eigendecomposition PCA uses now lives in internal/linalg.SymEig
// (shared with nn.GaLore).
