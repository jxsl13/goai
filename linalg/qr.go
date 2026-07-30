package linalg

import (
	"fmt"
	"github.com/jxsl13/goai/internal/parallel"
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
	rm := toRect(a, m, n) // working copy, becomes R (upper triangle)
	vs, betas := householder(rm, m, n)

	// R = the top n×n upper triangle
	rMat := make([]float64, n*n)
	for i := range n {
		for j := i; j < n; j++ {
			rMat[i*n+j] = rm[i][j]
		}
	}
	// reduced Q = H_0·H_1·…·H_{n−1} applied to the first n columns of I_m (apply reflectors in reverse)
	qcols := make([][]float64, m)
	for i := range m {
		qcols[i] = make([]float64, n)
		if i < n {
			qcols[i][i] = 1
		}
	}
	for k := n - 1; k >= 0; k-- {
		applyReflector(qcols, vs[k], betas[k], k, m, n)
	}
	qMat := make([]float64, m*n)
	for i := range m {
		copy(qMat[i*n:(i+1)*n], qcols[i])
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
	rm := toRect(a, m, n)
	vs, betas := householder(rm, m, n)
	for k := range n {
		if rm[k][k] == 0 {
			return nil, fmt.Errorf("linalg: Lstsq of a rank-deficient matrix (zero R[%d,%d])", k, k)
		}
	}
	vec := b.Ndim() == 1
	cols := 1
	if !vec {
		cols = b.Shape()[1]
	}
	out := make([]float64, n*cols) // [n,cols] row-major
	// Allocated ONCE for all columns rather than per column, matching what LU.Solve
	// already does. Safe because the buffer is fully overwritten at the start of its own
	// pass before anything reads it, so it cannot leak the previous column's values. It is
	// a function local, not a receiver field: that keeps concurrent calls independent
	// (PS6006 — a receiver slice used as per-call scratch is a data race waiting for its
	// second caller). Measured at n=512, cols=512: 1546 allocations per call down to 1035,
	// B/op 10.52MB down to 8.43MB.
	// PARALLEL over the right-hand-side columns, scratch per WORKER. Each column applies the
	// reflectors to its own Qᵀb copy and back-substitutes into its own out[i*cols+c], so the
	// columns are independent and the partition changes no value.
	solveCols(cols, m*n, m, func(clo, chi int, cvec []float64) {
		for c := clo; c < chi; c++ {
			for i := range m {
				if vec {
					cvec[i] = b.AtF64(i)
				} else {
					cvec[i] = b.AtF64(i, c)
				}
			}
			// Qᵀb: apply H_0,…,H_{n−1} in forward order (each reflector is symmetric)
			for k := range n {
				s := 0.0
				for i := k; i < m; i++ {
					s += vs[k][i] * cvec[i]
				}
				bt := betas[k] * s
				for i := k; i < m; i++ {
					cvec[i] -= bt * vs[k][i]
				}
			}
			// back-substitute R·x = (Qᵀb)[0:n]
			for i := n - 1; i >= 0; i-- {
				sum := cvec[i]
				for j := i + 1; j < n; j++ {
					sum -= rm[i][j] * out[j*cols+c]
				}
				out[i*cols+c] = sum / rm[i][i]
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
func householder(rm [][]float64, m, n int) (vs [][]float64, betas []float64) {
	vs = make([][]float64, n)
	betas = make([]float64, n)
	for k := range n {
		// x = rm[k:m, k]; norm below (incl.) the diagonal
		var norm float64
		for i := k; i < m; i++ {
			norm += rm[i][k] * rm[i][k]
		}
		norm = math.Sqrt(norm)
		v := make([]float64, m)
		vs[k] = v
		if norm == 0 {
			continue // column already zero → β=0, no reflection
		}
		// α = −sign(x₀)·‖x‖ (sign opposite to x₀ avoids cancellation in v₀)
		alpha := -norm
		if rm[k][k] < 0 {
			alpha = norm
		}
		for i := k; i < m; i++ {
			v[i] = rm[i][k]
		}
		v[k] -= alpha
		var vtv float64
		for i := k; i < m; i++ {
			vtv += v[i] * v[i]
		}
		if vtv == 0 {
			continue
		}
		beta := 2 / vtv
		betas[k] = beta
		// apply H_k to the trailing submatrix rm[k:m, k:n], PARALLEL over its columns. Column j
		// reads the reflector v — fixed for this step — and only its own rm[i][j], and writes
		// only its own rm[i][j]. So the columns are independent and the partition changes no
		// value: each dot product still accumulates over i ascending, and each element is
		// updated exactly once per k with k ascending.
		//
		// householder was 78% of Lstsq's wall clock at n=768 once the solve phase was
		// parallelized — the largest single serial block left in the package.
		//
		// The serial branch is written out rather than routed through the closure: this runs once
		// per reflector, n times per factorization, and a closure capturing step state costs one
		// allocation per call (PERF-CLOSURE-ON-SERIAL-BRANCH-001).
		cols := n - k
		if cols*(m-k) < factorParThreshold || parallel.Workers() <= 1 {
			for j := k; j < n; j++ {
				s := 0.0
				for i := k; i < m; i++ {
					s += v[i] * rm[i][j]
				}
				bs := beta * s
				for i := k; i < m; i++ {
					rm[i][j] -= bs * v[i]
				}
			}
		} else {
			parallelRowsIf(true, cols, func(lo, hi int) {
				for t := lo; t < hi; t++ {
					j := k + t
					s := 0.0
					for i := k; i < m; i++ {
						s += v[i] * rm[i][j]
					}
					bs := beta * s
					for i := k; i < m; i++ {
						rm[i][j] -= bs * v[i]
					}
				}
			})
		}
		rm[k][k] = alpha // exact diagonal
		for i := k + 1; i < m; i++ {
			rm[i][k] = 0 // annihilated below the diagonal
		}
	}
	return vs, betas
}

// applyReflector applies H_k = I − β·v·vᵀ to the rows k..m−1 of every column of the m×cols matrix q.
func applyReflector(q [][]float64, v []float64, beta float64, k, m, cols int) {
	if beta == 0 {
		return
	}
	// PARALLEL over columns. QR is sequential ACROSS reflectors — each depends on the
	// trailing submatrix the previous one left — but applying ONE reflector touches each
	// column independently: column j reads the shared read-only v and writes only q[i][j].
	//
	// BIT-IDENTICAL: the dot over i and the update over i are untouched within a column,
	// and no value crosses columns. A partition changes only which goroutine walks which
	// column.
	parallelCols(cols, m-k, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			s := 0.0
			for i := k; i < m; i++ {
				s += v[i] * q[i][j]
			}
			bs := beta * s
			for i := k; i < m; i++ {
				q[i][j] -= bs * v[i]
			}
		}
	})
}

// qrParThreshold is the total work (columns x rows) below which splitting the reflector
// application costs more than it saves — the same 1<<15 crossover measured for this class
// of core. QR peels one column per step, so the trailing submatrix shrinks and the later
// reflectors fall under it and run serially, which is correct.
const qrParThreshold = 1 << 15

// parallelCols splits cols across the shared bounded pool when there is enough work.
func parallelCols(cols, rows int, body func(lo, hi int)) {
	if cols*rows < qrParThreshold {
		body(0, cols)
		return
	}
	parallel.Rows(cols, body)
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

// toRect copies a into a fresh m×n [][]float64.
func toRect(a *tensor.Tensor, m, n int) [][]float64 {
	r := make([][]float64, m)
	for i := range m {
		r[i] = make([]float64, n)
		for j := range n {
			r[i][j] = a.AtF64(i, j)
		}
	}
	return r
}
