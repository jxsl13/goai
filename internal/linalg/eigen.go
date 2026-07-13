// Package linalg holds small dense-linear-algebra kernels shared across layers
// (classic PCA, nn GaLore) — currently a symmetric eigendecomposition. All f64.
package linalg

import "math"

// SymEig computes eigenvalues and eigenvectors of a symmetric matrix a via the
// cyclic Jacobi rotation method (Golub & Van Loan, Matrix Computations §8.5) —
// unconditionally stable for symmetric input. Eigenvalues are returned descending;
// vecs[k] is the eigenvector (length n) for vals[k]. a is not modified.
func SymEig(a [][]float64) (vals []float64, vecs [][]float64) {
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
