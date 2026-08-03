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
	// Working copy m and eigenvector accumulator v are FLAT [n*n] buffers, not slices of
	// rows. The Jacobi sweep below is O(n³); m is walked BOTH row-wise and column-wise so
	// it stays row-major, but v is only ever touched column-wise (the rotation and the
	// final extraction), so v is stored COLUMN-major (v[col*n+row]) — that turns its column
	// walk into a contiguous stream instead of a stride-n crawl, and makes the eigenvector
	// extraction a straight copy. Flat buffers also drop 2n allocations to 2. Index
	// arithmetic and storage location only: the operations, their order and their operands
	// are unchanged, so results are bit-identical. (m's remaining column walk cannot also be
	// made contiguous without a symmetry-exploiting rewrite that would change the arithmetic.)
	m := make([]float64, n*n)
	v := make([]float64, n*n)
	for i := range n {
		copy(m[i*n:i*n+n], a[i])
		v[i*n+i] = 1
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
				// v is stored COLUMN-major (v[col*n+row]), so columns p,q are two
				// contiguous slices — the rotation streams them instead of striding v by n
				// per k (as a row-major v[k*n+p] would). Identical values/order, bit-exact.
				vp := v[p*n : p*n+n]
				vq := v[q*n : q*n+n]
				for k := range n {
					vkp, vkq := vp[k], vq[k]
					vp[k] = c*vkp - s*vkq
					vq[k] = s*vkp + c*vkq
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
		col := make([]float64, n)
		copy(col, v[oi*n:oi*n+n]) // eigenvector oi is contiguous column oi (v is column-major)
		vecs[k] = col
	}
	return sorted, vecs
}
