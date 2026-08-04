package linalg

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// QR computes the reduced (economy) QR decomposition of an m×n matrix a with m ≥ n via Householder
// reflections (§R116; Golub & Van Loan §5.1-5.2, LAPACK dgeqrf): A = Q·R with Q ∈ R^{m×n} having
// orthonormal columns (QᵀQ = I_n) and R ∈ R^{n×n} upper-triangular. Householder QR is backward-stable
// (unlike classical Gram-Schmidt). Returns fresh f64 tensors; the input is not mutated.
func QR(a *tensor.Tensor) (q, r *tensor.Tensor, err error) {
	m, n, err := shapeMN(a)
	if err != nil {
		return nil, nil, err
	}
	if m < n {
		return nil, nil, fmt.Errorf("linalg: QR needs m ≥ n (tall/square), got %dx%d", m, n)
	}
	rm := toFlat(a, m, n) // row-major working copy, becomes R (upper triangle)
	t := make([]float64, n)
	vs, betas := householder(rm, m, n, t)

	// R = the top n×n upper triangle
	rMat := make([]float64, n*n)
	for i := range n {
		//perfscan:ignore PS4004 O(n2) R-copy inside O(mn2) factorization, PS4004 own note: no wall-clock
		for j := i; j < n; j++ {
			rMat[i*n+j] = rm[i*n+j]
		}
	}
	// reduced Q = H_0·H_1·…·H_{n−1} applied to the first n columns of I_m (apply reflectors in
	// reverse). qMat is the row-major [m,n] buffer, initialized to the first n columns of I_m.
	qMat := make([]float64, m*n)
	for i := range n {
		qMat[i*n+i] = 1
	}
	for k := n - 1; k >= 0; k-- {
		applyReflector(qMat, vs[k], betas[k], k, m, n, t)
	}
	return tensor.FromFloat64(tensor.Shape{m, n}, qMat), tensor.FromFloat64(tensor.Shape{n, n}, rMat), nil
}

// Lstsq solves the least-squares problem min‖A·x − b‖₂ for an m×n matrix a with m ≥ n (overdetermined
// or square) via Householder QR (§R116; LAPACK dgels): apply Qᵀ to b, then back-substitute
// R·x = (Qᵀb)[0:n]. More stable than the normal equations AᵀA·x = Aᵀb. b is a vector [m] or matrix
// [m,k]; x has shape [n] or [n,k]. Errors if R is rank-deficient (a zero on its diagonal).
func Lstsq(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	m, n, err := shapeMN(a)
	if err != nil {
		return nil, err
	}
	if m < n {
		return nil, fmt.Errorf("linalg: Lstsq needs m ≥ n, got %dx%d", m, n)
	}
	if b.Ndim() < 1 || b.Ndim() > 2 || b.Shape()[0] != m {
		return nil, fmt.Errorf("linalg: Lstsq rhs must be [%d] or [%d,k], got %v", m, m, b.Shape())
	}
	rm := toFlat(a, m, n)
	t := make([]float64, n)
	vs, betas := householder(rm, m, n, t)
	for k := range n {
		if rm[k*n+k] == 0 {
			return nil, fmt.Errorf("linalg: Lstsq of a rank-deficient matrix (zero R[%d,%d])", k, k)
		}
	}
	vec := b.Ndim() == 1
	cols := 1
	if !vec {
		cols = b.Shape()[1]
	}
	out := make([]float64, n*cols) // [n,cols] row-major
	// Read the right-hand side through its storage when it is already F64: AtF64 walks the shape
	// to a flat offset and dispatches on the storage type for every one of the m*cols elements.
	// Same values out of the same storage; any other dtype or layout keeps the accessor.
	var bf []float64
	if bc := b.Contiguous(); bc.Dtype() == tensor.F64 {
		bf = bc.Storage().F64()
	}
	// Right-hand sides are independent: column c reads the reflectors and R, and its own column of
	// b, and writes only out[·*cols+c]. Every column costs the same Qᵀb plus back substitution, so
	// equal bands need no balancing, and the buffer is allocated once per WORKER rather than once
	// per column — inside the band, so the fan-out does not give back the hoist.
	parallelCols(cols, m*n, func(lo, hi int) {
		//perfscan:ignore PS6008 per-worker cvec scratch, once per Lstsq call, GOMAXPROCS-bounded sanctioned
		cvec := make([]float64, m)
		//perfscan:ignore PS1006 per-column b gather, c is parallel outer, O(m) small share, no interchange
		for c := lo; c < hi; c++ {
			//perfscan:ignore PS1005 typed bf fast path already present, AtF64 is exotic-dtype fallback
			for i := range m {
				switch {
				case bf != nil && vec:
					cvec[i] = bf[i]
				case bf != nil:
					//perfscan:ignore PS6011 strided column gather into per-column scratch, O(m) small share
					cvec[i] = bf[i*cols+c]
				case vec:
					cvec[i] = b.AtF64(i)
				default:
					cvec[i] = b.AtF64(i, c)
				}
			}
			// Qᵀb: apply H_0,…,H_{n−1} in forward order (each reflector is symmetric)
			for k := range n {
				s := 0.0
				//perfscan:ignore PS4012 not quantized: beta is f64 reflector coeff; plain-dot covered by PS3010
				for i := k; i < m; i++ {
					s += vs[k][i] * cvec[i]
				}
				bt := betas[k] * s
				for i := k; i < m; i++ {
					//perfscan:ignore PS3075 memory-bound axpy (cvec streamed), bandwidth-bound; jam regresses
					cvec[i] -= bt * vs[k][i]
				}
			}
			// back-substitute R·x = (Qᵀb)[0:n]
			for i := n - 1; i >= 0; i-- {
				sum := cvec[i]
				for j := i + 1; j < n; j++ {
					//perfscan:ignore PS6011 back-sub recurrence, PS6011 own note: this Lstsq site measured -0.02% rejected
					sum -= rm[i*n+j] * out[j*cols+c]
				}
				out[i*cols+c] = sum / rm[i*n+i]
			}
		}
	})
	if vec {
		return tensor.FromFloat64(tensor.Shape{n}, out), nil
	}
	return tensor.FromFloat64(tensor.Shape{n, cols}, out), nil
}

// householder reduces the m×n matrix rm (in place) to upper-triangular R via Householder reflectors,
// returning the reflector vectors vs[k] (nonzero in rows k..m−1) and coefficients β_k such that
// H_k = I − β_k·v_k·v_kᵀ. R is left in rm's upper triangle.
func householder(rm []float64, m, n int, t []float64) (vs [][]float64, betas []float64) {
	vs = make([][]float64, n)
	betas = make([]float64, n)
	//perfscan:ignore PS1006,PS3059 column-norm reduction, strided column, no j-loop to interchange, O(mn) small | householder k-loop is sequentia
	for k := range n {
		// x = rm[k:m, k]; norm below (incl.) the diagonal
		var norm float64
		for i := k; i < m; i++ {
			//perfscan:ignore PS6011 column-norm read strided, unavoidable, O(m)/reflector small share
			x := rm[i*n+k]
			norm += x * x
		}
		norm = math.Sqrt(norm)
		//perfscan:ignore PS2008 reflector v alloc, PS2008 resource-only (no wall-clock per own note)
		v := make([]float64, m)
		vs[k] = v
		if norm == 0 {
			continue // column already zero → β=0, no reflection
		}
		// α = −sign(x₀)·‖x‖ (sign opposite to x₀ avoids cancellation in v₀)
		alpha := -norm
		if rm[k*n+k] < 0 {
			alpha = norm
		}
		//perfscan:ignore PS4004 false-positive: source rm[i*n+k] column-strided, no contiguous run, copy() inapplicable
		for i := k; i < m; i++ {
			//perfscan:ignore PS6011 strided column copy v[i]=rm[i*n+k], O(m)/reflector small share
			v[i] = rm[i*n+k]
		}
		v[k] -= alpha
		var vtv float64
		//perfscan:ignore PS3010 reflector norm dot O(m)/reflector, small vs O(mn2) rank-1 core; tol-risk in stable QR
		for i := k; i < m; i++ {
			vtv += v[i] * v[i]
		}
		if vtv == 0 {
			continue
		}
		beta := 2 / vtv
		betas[k] = beta
		// Apply H_k to the trailing submatrix rm[k:m, k:n] as a rank-1 update, i-OUTER so each
		// row segment rm[i*n+k:i*n+n] streams CONTIGUOUSLY (row-major) instead of the former
		// column-strided rm[i][j] scatter across m separate row slices. t[j] = Σ_i v[i]·rm[i,j]
		// in ascending i, scaled to β·t[j], then rm[i,j] -= (β·t[j])·v[i]. Same operands, same
		// ascending-i order, and the (β·s)·v[i] product grouping preserved → bit-identical.
		for j := k; j < n; j++ {
			t[j] = 0
		}
		for i := k; i < m; i++ {
			vi := v[i]
			row := rm[i*n : i*n+n]
			for j := k; j < n; j++ {
				t[j] += vi * row[j]
			}
		}
		for j := k; j < n; j++ {
			t[j] *= beta
		}
		for i := k; i < m; i++ {
			vi := v[i]
			row := rm[i*n : i*n+n]
			for j := k; j < n; j++ {
				row[j] -= t[j] * vi
			}
		}
		rm[k*n+k] = alpha // exact diagonal
		for i := k + 1; i < m; i++ {
			rm[i*n+k] = 0 // annihilated below the diagonal
		}
	}
	return vs, betas
}

// applyReflector applies H_k = I − β·v·vᵀ to the rows k..m−1 of every column of the m×cols matrix q.
func applyReflector(q []float64, v []float64, beta float64, k, m, cols int, t []float64) {
	if beta == 0 {
		return
	}
	// i-OUTER rank-1 update (see householder): stream each row q[i*cols:i*cols+cols]
	// contiguously instead of the column-strided q[i][j] scatter. Bit-identical.
	for j := range cols {
		t[j] = 0
	}
	for i := k; i < m; i++ {
		vi := v[i]
		row := q[i*cols : i*cols+cols]
		for j := range cols {
			t[j] += vi * row[j]
		}
	}
	for j := range cols {
		t[j] *= beta
	}
	for i := k; i < m; i++ {
		vi := v[i]
		row := q[i*cols : i*cols+cols]
		for j := range cols {
			row[j] -= t[j] * vi
		}
	}
}

// shapeMN validates a rank-2 non-empty matrix and returns its dimensions.
func shapeMN(a *tensor.Tensor) (m, n int, err error) {
	if a.Ndim() != 2 {
		return 0, 0, fmt.Errorf("linalg: expected a rank-2 matrix, got %v", a.Shape())
	}
	m, n = a.Shape()[0], a.Shape()[1]
	if m == 0 || n == 0 {
		return 0, 0, fmt.Errorf("linalg: empty matrix %v", a.Shape())
	}
	return m, n, nil
}

// toFlat copies a into a fresh m×n row-major []float64 (one contiguous allocation).
func toFlat(a *tensor.Tensor, m, n int) []float64 {
	r := make([]float64, m*n)
	for i := range m {
		//perfscan:ignore PS1005 one-time O(mn) toFlat input flatten, small share vs O(mn2) factorization
		for j := range n {
			r[i*n+j] = a.AtF64(i, j)
		}
	}
	return r
}
