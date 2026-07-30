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
				// Six bounds checks per k survived across these three loops, confirmed
				// with -d=ssa/check_bce/debug=1. Note that the row-slice hoist below did
				// NOT remove its own pair: `for k := range n` gives prove no relation
				// between n and len(rp), so hoisting the slice is not by itself enough.
				//
				// The column loop cannot use row slices — it walks a column — so it runs
				// on two induction offsets against one hoisted bound instead. The row and
				// accumulator loops range over one slice with the other clamped to the
				// same length, which is the shape prove can discharge.
				//
				// Bit-identical: every destination is written once from the same two
				// source values, in the same k order. Only branches are removed.
				// v is stored TRANSPOSED (vT[p*n+k] is the old v[k*n+p]). The
				// accumulator is only ever column-rotated, so transposing its storage
				// turns the walk contiguous without touching the arithmetic — same
				// operations, same order, same operands (PS6011).
				rp := vT[p*n : p*n+n : p*n+n]
				rq := vT[q*n : q*n+n : q*n+n]
				rq = rq[:len(rp)]
				// THE COLUMN ROTATION OF m AND THE ROTATION OF THE ACCUMULATOR ARE FUSED.
				// They are disjoint by construction — one touches columns p and q of m, the
				// other touches only vT — so no peel and no skip is needed and the two
				// dependency chains issue together. Fusing the ROW rotation in as well would
				// need a four-element peel, because at k in {p,q} the row loop reads cells the
				// column loop has just written; that is left separate.
				//
				// The loop ranges over rp with rq clamped to it, which is what discharges
				// their bounds checks — ranging over a separate count does not, since prove
				// gets no relation between the count and the slice length. m keeps its two
				// computed-offset checks, as before.
				//
				// Bit-identical: the column rotation still completes for every k before the
				// row rotation begins, every destination is written once from the same two
				// source values, and the k order is unchanged in all three.
				for k, vkp := range rp {
					kp, kq := k*n+p, k*n+q
					mkp, mkq := m[kp], m[kq]
					m[kp] = c*mkp - s*mkq
					m[kq] = s*mkp + c*mkq
					vkq := rq[k]
					rp[k] = c*vkp - s*vkq
					rq[k] = s*vkp + c*vkq
				}
				rmp := m[p*n : p*n+n : p*n+n]
				rmq := m[q*n : q*n+n : q*n+n]
				rmq = rmq[:len(rmp)]
				for k, mpk := range rmp {
					mqk := rmq[k]
					rmp[k] = c*mpk - s*mqk
					rmq[k] = s*mpk + c*mqk
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
