// Package linalg holds small dense-linear-algebra kernels shared across layers
// (classic PCA, nn GaLore/Shampoo/SOAP) — currently a symmetric eigendecomposition. All f64.
package linalg

import "math"

// eigEps is the unit roundoff (2^-52), the subdiagonal-negligibility threshold in the QL sweep.
const eigEps = 2.220446049250313e-16

// SymEig computes eigenvalues and eigenvectors of a symmetric matrix a via Householder
// tridiagonalization followed by the implicit-shift symmetric QL algorithm (Golub & Van
// Loan, Matrix Computations §8.3; EISPACK tred2/tql2). Eigenvalues are returned descending;
// vecs[k] is the eigenvector (length n) for vals[k]. a is not modified.
//
// This replaces the earlier cyclic-Jacobi solver: tridiag+QL converges in a BOUNDED number
// of QL sweeps (Wilkinson shift → cubic convergence) instead of Jacobi's O(sweeps·n³) that
// blows up on clustered spectra (a synthetic n=512 clustered matrix took ~55 s under Jacobi).
// It is backward stable with error O(ε‖A‖) — the same order as Jacobi and the same algorithm
// family as the LAPACK dsyev/sklearn reference the PCA parity test checks against. Not
// bit-identical to Jacobi; callers rely only on the numeric contract (see eigen_bitident_test.go,
// which cross-checks this against the retained Jacobi oracle to ~1e-9 eigenvalues / 1e-10
// reconstruction rather than bit-for-bit).
func SymEig(a [][]float64) (vals []float64, vecs [][]float64) {
	n := len(a)
	if n == 0 {
		return nil, nil
	}
	// z is the flat [n*n] row-major working matrix: it starts as a copy of a (so a is not
	// modified), tred2 overwrites it with the accumulated Householder Q, and tql2 rotates
	// it into the eigenvector matrix — column c of z is the eigenvector for d[c].
	z := make([]float64, n*n)
	for i := range n {
		copy(z[i*n:i*n+n], a[i])
	}
	d := make([]float64, n) // diagonal / eigenvalues
	e := make([]float64, n) // subdiagonal
	tred2(z, n, d, e)
	// Transpose the Householder Q to COLUMN-major (zc[c*n+r] = column c of Q) before the QL
	// sweep. tql2's dominant O(n³) cost is the per-rotation eigenvector update, which touches
	// two logical COLUMNS across all n rows — column-strided (stride n) in row-major z, but two
	// contiguous slices in zc. The O(n²) transpose is negligible against the O(n³) sweep.
	// Storage layout only: the QL arithmetic, its order and operands are unchanged (bit-identical).
	zc := make([]float64, n*n)
	for r := range n {
		zr := z[r*n : r*n+n]
		for c := range n {
			zc[c*n+r] = zr[c]
		}
	}
	tql2(zc, n, d, e)

	// extract + sort descending (same output layout the callers expect).
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	for i := range n { // selection sort (n small); largest eigenvalue first
		best := i
		for j := i + 1; j < n; j++ {
			if d[order[j]] > d[order[best]] {
				best = j
			}
		}
		order[i], order[best] = order[best], order[i]
	}
	vals = make([]float64, n)
	vecs = make([][]float64, n) // vecs[k] = k-th eigenvector
	for k, oi := range order {
		vals[k] = d[oi]
		col := make([]float64, n)
		copy(col, zc[oi*n:oi*n+n]) // eigenvector oi is column oi of Q = contiguous row oi of zc
		vecs[k] = col
	}
	return vals, vecs
}

// tred2 reduces the symmetric matrix z (flat [n*n] row-major) to tridiagonal form by
// Householder transformations, accumulating the orthogonal reduction matrix Q back into z.
// On return d holds the diagonal, e[1..n-1] the subdiagonal (e[0]=0), and z = Q. This is
// the classic EISPACK tred2 / Numerical Recipes §11.2, 0-indexed. (Golub & Van Loan §8.3.1.)
func tred2(z []float64, n int, d, e []float64) {
	for i := n - 1; i > 0; i-- {
		l := i - 1
		h, scale := 0.0, 0.0
		if l > 0 {
			for k := 0; k <= l; k++ {
				scale += math.Abs(z[i*n+k])
			}
			if scale == 0.0 {
				e[i] = z[i*n+l] // skip transformation
			} else {
				for k := 0; k <= l; k++ {
					z[i*n+k] /= scale
					h += z[i*n+k] * z[i*n+k]
				}
				f := z[i*n+l]
				g := math.Sqrt(h)
				if f >= 0 {
					g = -g
				}
				e[i] = scale * g
				h -= f * g
				z[i*n+l] = f - g
				f = 0.0
				for j := 0; j <= l; j++ {
					z[j*n+i] = z[i*n+j] / h // store u/h in column i for the Q accumulation
					g = 0.0
					for k := 0; k <= j; k++ {
						g += z[j*n+k] * z[i*n+k]
					}
					for k := j + 1; k <= l; k++ {
						g += z[k*n+j] * z[i*n+k]
					}
					e[j] = g / h // p, provisionally in e
					f += e[j] * z[i*n+j]
				}
				hh := f / (h + h)
				for j := 0; j <= l; j++ {
					f = z[i*n+j]
					g = e[j] - hh*f // q = p - K·u
					e[j] = g
					for k := 0; k <= j; k++ {
						z[j*n+k] -= f*e[k] + g*z[i*n+k] // rank-2 update of the trailing block
					}
				}
			}
		} else {
			e[i] = z[i*n+l]
		}
		d[i] = h
	}
	d[0] = 0.0
	e[0] = 0.0
	// accumulate the transformation matrices into z (eigenvectors of the reflectors).
	for i := 0; i < n; i++ {
		l := i - 1
		if d[i] != 0.0 {
			for j := 0; j <= l; j++ {
				g := 0.0
				for k := 0; k <= l; k++ {
					g += z[i*n+k] * z[k*n+j]
				}
				for k := 0; k <= l; k++ {
					z[k*n+j] -= g * z[k*n+i]
				}
			}
		}
		d[i] = z[i*n+i]
		z[i*n+i] = 1.0
		for j := 0; j <= l; j++ {
			z[j*n+i] = 0.0
			z[i*n+j] = 0.0
		}
	}
}

// tql2 diagonalizes the symmetric tridiagonal matrix (diagonal d, subdiagonal e) by the
// implicit-shift QL algorithm with Wilkinson shifts, accumulating the plane rotations into
// the eigenvector matrix z (which enters as the Householder Q from tred2, stored COLUMN-major:
// z[c*n+r] is row r of column c). On return d holds the eigenvalues (unordered) and column c
// of z (the contiguous slice z[c*n:(c+1)*n]) is the eigenvector for d[c]. EISPACK tql2 /
// Numerical Recipes §11.3, 0-indexed. (Golub & Van Loan §8.3.3.)
func tql2(z []float64, n int, d, e []float64) {
	if n == 1 {
		return
	}
	for i := 1; i < n; i++ { // renumber e so the subdiagonal is e[0..n-2]
		e[i-1] = e[i]
	}
	e[n-1] = 0.0
	for l := 0; l < n; l++ {
		iter := 0
		for {
			// find the smallest m ≥ l with a negligible subdiagonal e[m]
			m := l
			for ; m < n-1; m++ {
				dd := math.Abs(d[m]) + math.Abs(d[m+1])
				if math.Abs(e[m]) <= eigEps*dd {
					break
				}
			}
			if m == l {
				break // d[l] has converged
			}
			iter++
			if iter > 50 {
				break // safeguard: QL on symmetric tridiagonals always converges well within this
			}
			// Wilkinson shift from the trailing 2×2.
			g := (d[l+1] - d[l]) / (2.0 * e[l])
			r := math.Hypot(g, 1.0)
			g = d[m] - d[l] + e[l]/(g+math.Copysign(r, g))
			s, c := 1.0, 1.0
			p := 0.0
			broke := false
			for i := m - 1; i >= l; i-- { // chase the bulge, Givens rotations
				f := s * e[i]
				b := c * e[i]
				r = math.Hypot(f, g)
				e[i+1] = r
				if r == 0.0 {
					d[i+1] -= p
					e[m] = 0.0
					broke = true
					break
				}
				s = f / r
				c = g / r
				g = d[i+1] - p
				r = (d[i]-g)*s + 2.0*c*b
				p = s * r
				d[i+1] = g + p
				g = c*r - b
				// z is COLUMN-major here, so eigenvector columns i and i+1 are two contiguous
				// slices — the rotation streams them instead of striding z by n per row k.
				ci := z[i*n : i*n+n]
				ci1 := z[(i+1)*n : (i+1)*n+n]
				for k := 0; k < n; k++ { // accumulate the rotation into eigenvector columns i, i+1
					f = ci1[k]
					ci1[k] = s*ci[k] + c*f
					ci[k] = c*ci[k] - s*f
				}
			}
			if broke {
				continue
			}
			d[l] -= p
			e[l] = g
			e[m] = 0.0
		}
	}
}
