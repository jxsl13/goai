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
	// Working copy and eigenvector accumulator are FLAT [n*n] row-major buffers, not
	// slices of rows. The Jacobi sweep below is O(n³) and two of its three inner
	// loops walk a COLUMN (m[k][p], v[k][p]) — with a row-of-slices layout every k
	// dereferences a different, independently allocated row, which is the worst
	// access pattern that layout has. Flat buffers make those walks a constant stride
	// through one allocation and drop 2n allocations to 2. Index arithmetic only: the
	// operations, their order and their operands are unchanged, so results are
	// bit-identical.
	m := make([]float64, n*n)
	vT := make([]float64, n*n) // TRANSPOSED: vT[c*n+r] is eigenvector-column c, row r
	for i := range n {
		copy(m[i*n:i*n+n], a[i])
		vT[i*n+i] = 1 // identity is its own transpose
	}
	for range 100 {
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				off += m[i*n+j] * m[i*n+j]
			}
		}
		if off < 1e-28 {
			break
		}
		for p := range n {
			for q := p + 1; q < n; q++ {
				if math.Abs(m[p*n+q]) < 1e-300 {
					continue
				}
				// Jacobi rotation zeroing m[p][q]
				theta := (m[q*n+q] - m[p*n+p]) / (2 * m[p*n+q])
				t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := range n {
					mkp, mkq := m[k*n+p], m[k*n+q]
					m[k*n+p] = c*mkp - s*mkq
					m[k*n+q] = s*mkp + c*mkq
				}
				for k := range n {
					mpk, mqk := m[p*n+k], m[q*n+k]
					m[p*n+k] = c*mpk - s*mqk
					m[q*n+k] = s*mpk + c*mqk
				}
				// v is stored TRANSPOSED (vT[p*n+k] is the old v[k*n+p]). The
				// accumulator is only ever column-rotated, so transposing its storage
				// turns the walk contiguous without touching the arithmetic — same
				// operations, same order, same operands (PS6011).
				rp := vT[p*n : p*n+n : p*n+n]
				rq := vT[q*n : q*n+n : q*n+n]
				for k := range n {
					vkp, vkq := rp[k], rq[k]
					rp[k] = c*vkp - s*vkq
					rq[k] = s*vkp + c*vkq
				}
			}
		}
	}
	// extract + sort descending
	vals = make([]float64, n)
	for i := range n {
		vals[i] = m[i*n+i]
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
		// The eigenvector is one COLUMN of v, which in transposed storage is a
		// contiguous row — the extraction was a strided walk too.
		col := make([]float64, n)
		copy(col, vT[oi*n:oi*n+n])
		vecs[k] = col
	}
	return sorted, vecs
}
