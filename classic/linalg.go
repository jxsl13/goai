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

// jacobiEigen computes eigenvalues and eigenvectors of a symmetric matrix via
// the cyclic Jacobi rotation method (Golub & Van Loan §8.5) — unconditionally
// stable for symmetric input. Returns eigenvalues descending with matching
// eigenvector columns.
func jacobiEigen(a [][]float64) (vals []float64, vecs [][]float64) {
	n := len(a)
	// working copy + identity eigenvector accumulator
	m := make([][]float64, n)
	v := make([][]float64, n)
	for i := range n {
		m[i] = append([]float64(nil), a[i]...)
		v[i] = make([]float64, n)
		v[i][i] = 1
	}
	for range 100 {
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				off += m[i][j] * m[i][j]
			}
		}
		if off < 1e-28 {
			break
		}
		for p := range n {
			for q := p + 1; q < n; q++ {
				if math.Abs(m[p][q]) < 1e-300 {
					continue
				}
				// Jacobi rotation zeroing m[p][q]
				theta := (m[q][q] - m[p][p]) / (2 * m[p][q])
				t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := range n {
					mkp, mkq := m[k][p], m[k][q]
					m[k][p] = c*mkp - s*mkq
					m[k][q] = s*mkp + c*mkq
				}
				for k := range n {
					mpk, mqk := m[p][k], m[q][k]
					m[p][k] = c*mpk - s*mqk
					m[q][k] = s*mpk + c*mqk
				}
				for k := range n {
					vkp, vkq := v[k][p], v[k][q]
					v[k][p] = c*vkp - s*vkq
					v[k][q] = s*vkp + c*vkq
				}
			}
		}
	}
	// extract + sort descending
	vals = make([]float64, n)
	for i := range n {
		vals[i] = m[i][i]
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	for i := range n { // simple selection sort (n tiny)
		best := i
		for j := i + 1; j < n; j++ {
			if vals[order[j]] > vals[order[best]] {
				best = j
			}
		}
		order[i], order[best] = order[best], order[i]
	}
	sorted := make([]float64, n)
	vecs = make([][]float64, n) // vecs[k] = k-th eigenvector
	for k, oi := range order {
		sorted[k] = vals[oi]
		col := make([]float64, n)
		for r := range n {
			col[r] = v[r][oi]
		}
		vecs[k] = col
	}
	return sorted, vecs
}
