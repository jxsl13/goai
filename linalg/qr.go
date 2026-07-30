package linalg

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/internal/parallel"
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
	// Each row's kept span is the contiguous run [i, n), so the elementwise loop was a memmove
	// written out by hand (PS1008). copy() is 2.35x-3.16x faster than the loop on this shape; the
	// compiler does not recognize the ARR[i*n+j] form. rMat is freshly allocated and rm is a
	// separate buffer, so the two never alias.
	for i := range n {
		copy(rMat[i*n+i:i*n+n], rm[i*n+i:i*n+n])
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
	// PARALLEL over right-hand-side columns, the same fan-out LU.Solve already uses. Column c
	// reads vs/betas/rm — all read-only once the factorization is done — and writes only
	// out[·*cols+c], with its own cvec; nothing crosses columns, and within a column the Qᵀb
	// application and the back substitution keep their exact order, so the partition is
	// bit-identical.
	//
	// This loop was the last serial block here and it is the dominant one: at GOMAXPROCS=1 the
	// three statements inside it (the Qᵀb dot, the Qᵀb update, and the back-substitution inner
	// product) are 68% of Lstsq, against 17% for the householder factorization. The scaling curve
	// said so before the profile did — LstsqMat/768 ran 830ms on one core and 792ms on twelve,
	// 1.05x, the worst scaler of any benchmark in this package.
	//
	// cvec is the per-WORKER scratch solveCols hands out rather than a per-column make(): it is
	// fully rewritten at the top of every column, so reuse across a worker's columns is safe.
	solveCols(cols, n*n, m, func(clo, chi int, cvec []float64) {
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
	// ONE slab for all n reflector vectors instead of one make() per column. The bytes are the
	// same; the allocation COUNT drops from n to 1, which is what the allocator was paying for —
	// at 768x768 this call site alone issued 768 separate 6 KB allocations, and a profile of
	// Lstsq showed 24.4% of samples inside runtime.madvise, the scavenger returning and
	// re-faulting those spans. The views are disjoint (column k owns [k*m, (k+1)*m)) and each is
	// capped, so nothing can append across a boundary; vs[k] is only read after construction.
	slab := make([]float64, n*m)
	for k := range n {
		// x = rm[k:m, k]; norm below (incl.) the diagonal
		var norm float64
		for i := k; i < m; i++ {
			x := rm[i*n+k]
			norm += x * x
		}
		norm = math.Sqrt(norm)
		v := slab[k*m : k*m+m : k*m+m]
		vs[k] = v
		if norm == 0 {
			continue // column already zero → β=0, no reflection
		}
		// α = −sign(x₀)·‖x‖ (sign opposite to x₀ avoids cancellation in v₀)
		alpha := -norm
		if rm[k*n+k] < 0 {
			alpha = norm
		}
		for i := k; i < m; i++ {
			v[i] = rm[i*n+k]
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
		// Apply H_k to the trailing submatrix rm[k:m, k:n] as a rank-1 update, i-OUTER so each
		// row segment rm[i*n+k:i*n+n] streams CONTIGUOUSLY (row-major) instead of the former
		// column-strided rm[i][j] scatter across m separate row slices. t[j] = Σ_i v[i]·rm[i,j]
		// in ascending i, scaled to β·t[j], then rm[i,j] -= (β·t[j])·v[i]. Same operands, same
		// ascending-i order, and the (β·s)·v[i] product grouping preserved → bit-identical.
		for j := k; j < n; j++ {
			t[j] = 0
		}
		// The i loop is unrolled by 2 with SEPARATE accumulating adds, so t[j] stays in a
		// register across the pair instead of being loaded and stored once per row. Both rows
		// are still read contiguously. Bit-identical: the two adds keep i ascending and each
		// term keeps its own vi*row[j] association, so nothing is reassociated.
		//
		// The alternative PS1007 names first — strip-mine the j loop by 4 and put i innermost —
		// was MEASURED and is worse here: it trades one contiguous pass over the submatrix for
		// n/4 strided passes, and its gain DECAYS with the outer trip count (LstsqMat -2.56% at
		// n=64, -1.92% at 256, -0.52% at 512, indistinguishable from base at 768). This form
		// holds -2.2% to -3.0% at every size. Strip-mining is the right remedy only when the
		// input is NOT already contiguous in the inner var.
		i := k
		for ; i+2 <= m; i += 2 {
			vi0, vi1 := v[i], v[i+1]
			r0 := rm[i*n : i*n+n]
			r1 := rm[(i+1)*n : (i+1)*n+n]
			for j := k; j < n; j++ {
				t[j] += vi0 * r0[j]
				t[j] += vi1 * r1[j]
			}
		}
		for ; i < m; i++ {
			vi := v[i]
			row := rm[i*n : i*n+n]
			for j := k; j < n; j++ {
				t[j] += vi * row[j]
			}
		}
		for j := k; j < n; j++ {
			t[j] *= beta
		}
		// The APPLY phase parallelizes over rows; the ACCUMULATE above cannot, and the
		// asymmetry is the whole point. Accumulating t[j] = Σ_i v[i]·rm[i,j] is a reduction
		// over i, so splitting i across workers would need per-worker partials and a merge —
		// that reassociates the sum and is not bit-identical. Here row i reads only t (fixed
		// for this reflector) and its own row segment, and writes only that segment: the rows
		// are independent, every element is still updated exactly once per k with k ascending,
		// and no value crosses a chunk. So this half is bit-identical under any partition and
		// the other half stays serial and contiguous.
		//
		// The serial branch is written out rather than routed through the closure: this runs
		// once per reflector, n times per factorization, and a closure capturing step state
		// costs an allocation per call (PERF-CLOSURE-ON-SERIAL-BRANCH-001).
		if rows := m - k; (n-k)*rows < factorParThreshold || parallel.Workers() <= 1 {
			for i := k; i < m; i++ {
				vi := v[i]
				row := rm[i*n : i*n+n]
				for j := k; j < n; j++ {
					row[j] -= t[j] * vi
				}
			}
		} else {
			parallel.Rows(rows, func(lo, hi int) {
				for i := k + lo; i < k+hi; i++ {
					vi := v[i]
					row := rm[i*n : i*n+n]
					for j := k; j < n; j++ {
						row[j] -= t[j] * vi
					}
				}
			})
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
	// Unrolled by 2 over i, as in householder — see the note there for why strip-mining the
	// j loop instead measured worse.
	i := k
	for ; i+2 <= m; i += 2 {
		vi0, vi1 := v[i], v[i+1]
		r0 := q[i*cols : i*cols+cols]
		r1 := q[(i+1)*cols : (i+1)*cols+cols]
		for j := range cols {
			t[j] += vi0 * r0[j]
			t[j] += vi1 * r1[j]
		}
	}
	for ; i < m; i++ {
		vi := v[i]
		row := q[i*cols : i*cols+cols]
		for j := range cols {
			t[j] += vi * row[j]
		}
	}
	for j := range cols {
		t[j] *= beta
	}
	// Rows are independent here for the same reason as in householder — see the note there.
	if rows := m - k; cols*rows < factorParThreshold || parallel.Workers() <= 1 {
		for i := k; i < m; i++ {
			vi := v[i]
			row := q[i*cols : i*cols+cols]
			for j := range cols {
				row[j] -= t[j] * vi
			}
		}
	} else {
		parallel.Rows(rows, func(lo, hi int) {
			for i := k + lo; i < k+hi; i++ {
				vi := v[i]
				row := q[i*cols : i*cols+cols]
				for j := range cols {
					row[j] -= t[j] * vi
				}
			}
		})
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
	// A contiguous row-major f64 input is already in the target layout, so the whole matrix is
	// one copy instead of m*n AtF64 dispatches — 590k of them at 768x768 (PS1001). The COPY is
	// load-bearing and must not become an alias: householder mutates this buffer in place, and
	// both QR and Lstsq document that the input tensor is left untouched.
	if src, ok := flatRowMajor(a); ok {
		copy(r, src[:m*n])
		return r
	}
	for i := range m {
		for j := range n {
			r[i*n+j] = a.AtF64(i, j)
		}
	}
	return r
}
