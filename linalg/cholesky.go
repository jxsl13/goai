package linalg

import (
	"fmt"
	"github.com/jxsl13/goai/internal/parallel"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Cholesky returns the lower-triangular Cholesky factor L of a real symmetric positive-definite (SPD)
// matrix, A = L·Lᵀ with L[i,i] > 0 (numpy.linalg.cholesky / LAPACK dpotrf, §R120). It uses the
// Cholesky-Banachiewicz recurrence (no pivoting — SPD needs none) and returns an error exactly when A
// is not positive definite (a non-positive √ argument at some leading minor). a must be square and
// symmetric; it is not mutated. Cholesky is ~2× faster than LU and is the standard direct solver for
// SPD systems (normal equations, covariance, Gaussian processes).
func Cholesky(a *tensor.Tensor) (*tensor.Tensor, error) {
	n, err := squareN(a)
	if err != nil {
		return nil, err
	}
	if err := requireSymmetric(a, n); err != nil {
		return nil, err
	}
	l, err := cholFactor(a, n)
	if err != nil {
		return nil, err
	}
	out := make([]float64, n*n)
	// Row i's lower triangle is the contiguous run [0, i+1) on both sides (PS1008), so this is one
	// memmove per row rather than i+1 bounds-checked stores: 2.27x-3.46x on this shape. out is fresh
	// and l is the factor's own row storage, so there is no aliasing.
	for i := range n {
		copy(out[i*n:i*n+i+1], l[i][:i+1])
	}
	return tensor.FromFloat64(tensor.Shape{n, n}, out), nil
}

// CholSolve solves the SPD system A·X = B via the Cholesky factorization (LAPACK dpotrs): forward-
// substitution L·Y = B, then back-substitution Lᵀ·X = Y. B is a vector [n] or matrix [n,k]; errors if
// A is not SPD.
func CholSolve(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	n, err := squareN(a)
	if err != nil {
		return nil, err
	}
	if b.Ndim() < 1 || b.Ndim() > 2 || b.Shape()[0] != n {
		return nil, fmt.Errorf("linalg: CholSolve rhs must be [%d] or [%d,k], got %v", n, n, b.Shape())
	}
	if err := requireSymmetric(a, n); err != nil {
		return nil, err
	}
	l, err := cholFactor(a, n)
	if err != nil {
		return nil, err
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
	// second caller). Measured at n=512, cols=512: 1032 allocations per call down to 521,
	// B/op 8.40MB down to 6.31MB.
	// PARALLEL over the right-hand-side columns, with the scratch per WORKER. Columns are
	// independent — each writes only its own out[i*cols+c] and reads only its own pass's y — so
	// the partition changes no value. Hoisting one shared buffer had coupled them.
	// The forward pass reads one rhs element per (column, row), so n*cols interface dispatches
	// per solve — 262k at n=512, cols=512 (PS1005). bf is the contiguous f64 backing slice when
	// there is one, covering both the vector and matrix shapes; nil falls back to AtF64.
	bf, _ := flatContig(b)
	solveCols(cols, n*n, n, func(clo, chi int, y []float64) {
		for c := clo; c < chi; c++ {
			for i := range n { // forward: L·y = b
				var s float64
				switch {
				case bf != nil && vec:
					s = bf[i]
				case bf != nil:
					s = bf[i*cols+c]
				case vec:
					s = b.AtF64(i)
				default:
					s = b.AtF64(i, c)
				}
				// RANGE OVER THE ROW, not over the index. l[i][k] re-loads the row pointer and
				// bounds-checks against it every step; ranging li[:i] hoists the pointer and
				// lets the compiler drop the check, which is the half that actually pays —
				// hoisting alone, keeping `for k := range i`, measured a wash here.
				// Bit-identical: same operands, same ascending-k order.
				li := l[i]
				yr := y[:i]
				for k, lik := range li[:i] {
					s -= lik * yr[k]
				}
				y[i] = s / li[i]
			}
			// Back-substitution accumulates in the CONTIGUOUS scratch, not through out — the same
			// transform LU.Solve already carries, and for the same reason: reading out[k*cols+c]
			// with c fixed jumps a whole output row per step, 4 KB at n=cols=512, touching a fresh
			// cache line to use eight bytes of it. A line-level profile put the equivalent
			// statement in LU.Solve at 32% of BenchmarkInverse. The values read are the x[k] this
			// same loop wrote on earlier iterations, so they can live in y: x[i] overwrites y[i]
			// only AFTER y[i] has been read into s, and the forward pass's y[k] for k > i was
			// already consumed at step k, so nothing still needed is clobbered.
			//
			// Bit-identical: same operands, same descending-i order, same ascending-k accumulation
			// into s, same final divide. Only where the intermediate lives changes.
			//
			// The l[k][i] stream is still column-strided — l is row-major, so stepping k with i
			// fixed walks a column. TRANSPOSING L ONCE WAS TRIED AND MEASURED AND DOES NOT PAY, so
			// this reads the column deliberately rather than for want of the idea.
			//
			// The arithmetic looked convincing: n*n/2 writes once against n*n*cols/2 strided reads
			// saved, a 1/cols overhead, 0.2% at cols=512. Measured over 18 samples per arm,
			// interleaved: CholSolveMat/64 +3.74% (p=0.000), /256 -0.63% (p=0.031), /512 and /768
			// indistinguishable; Inverse/64 -2.23%, its other three sizes indistinguishable. A wash
			// with a real regression at the small end, where the extra n*n allocation and the full
			// pass over L cost more than a stride the prefetcher was already handling.
			//
			// perfscan PS1010 reports this line, correctly by its own predicate — the check finds
			// the shape, not the payoff, which is why its output says candidates rather than wins.
			for i := n - 1; i >= 0; i-- { // back: Lᵀ·x = y (Lᵀ[i,k] = L[k,i])
				s := y[i]
				// Three bounds checks against ONE multiply-add: l[k], then [i], then y[k].
				// Ranging the ROW POINTERS and pairing y to the same extent leaves only lk[i],
				// taking the inner loop from 3 checks to 1 (verified with
				// -gcflags=-d=ssa/check_bce/debug=1). There is no row to range along here — the
				// walk is down a column — but the slice of rows can be, which is the part the
				// earlier column-gather rejection did not cover.
				//
				// Bit-identical: lr[t] is l[i+1+t] and yr[t] is y[i+1+t], the same operands in the
				// same ascending order.
				lr := l[i+1 : n]
				yr := y[i+1 : n]
				yr = yr[:len(lr)]
				for t, lk := range lr {
					s -= lk[i] * yr[t]
				}
				y[i] = s / l[i][i]
			}
			for i := range n {
				out[i*cols+c] = y[i]
			}
		}
	})
	if vec {
		return tensor.FromFloat64(tensor.Shape{n}, out), nil
	}
	return tensor.FromFloat64(tensor.Shape{n, cols}, out), nil
}

// cholFactor computes the lower-triangular Cholesky factor L (as [][]float64) of the n×n SPD matrix a,
// erroring if a is not positive definite.
func cholFactor(a *tensor.Tensor, n int) ([][]float64, error) {
	// ONE slab for the n rows of L, not one make() per row: at n=768 that is 768 separate
	// allocations per factorization replaced by one. Rows are disjoint capped views, they are
	// written in place rather than replaced, and nothing appends across a boundary. Bit-identical
	// — make() zeroes and so does a fresh slab (PS2008).
	//
	// Allocation count only: the two prior measurements of this transform (QR reflectors, GMM
	// responsibilities) both moved the wall clock not at all, so no latency claim is made here.
	l := make([][]float64, n)
	lslab := make([]float64, n*n)
	for i := range n {
		l[i] = lslab[i*n : i*n+n : i*n+n]
	}
	// One contiguous view of a for the whole factorization: the column update below reads
	// a[i,j] for every i>j, so the naive walk costs n²/2 interface dispatches per factorization
	// on top of the n diagonal reads (PS1005). af == nil keeps the AtF64 path for non-f64 or
	// strided inputs. Same values in the same order either way.
	af, afOK := flatRowMajor(a)
	atA := func(i, j int) float64 {
		if afOK {
			return af[i*n+j]
		}
		return a.AtF64(i, j)
	}
	for j := range n {
		d := atA(j, j)
		for k := range j {
			d -= l[j][k] * l[j][k]
		}
		if d <= 0 {
			return nil, fmt.Errorf("linalg: matrix is not positive definite (non-positive pivot %g at leading minor %d)", d, j)
		}
		l[j][j] = math.Sqrt(d)
		// PARALLEL over the rows below the diagonal. Row i reads its OWN already-computed columns
		// l[i][k] for k < j and the fixed row l[j][k], and writes only l[i][j] — so the rows are
		// independent within column j and the partition changes no value. Each subtraction still
		// runs k ascending.
		//
		// Column j must complete before j+1 begins, which is why the outer loop stays serial; the
		// serial branch is written out inline because this runs once per column (n times per
		// factorization) and a closure would allocate per call.
		rows := n - j - 1
		ljj := l[j][j]
		lj := l[j]
		if rows*j < factorParThreshold || parallel.Workers() <= 1 {
			for i := j + 1; i < n; i++ {
				s := atA(i, j)
				li := l[i]
				for k := range j {
					s -= li[k] * lj[k]
				}
				li[j] = s / ljj
			}
		} else {
			parallelRowsIf(true, rows, func(lo, hi int) {
				for t := lo; t < hi; t++ {
					i := j + 1 + t
					s := atA(i, j)
					li := l[i]
					for k := range j {
						s -= li[k] * lj[k]
					}
					li[j] = s / ljj
				}
			})
		}
	}
	return l, nil
}

// requireSymmetric errors if a is not symmetric to a relative tolerance.
func requireSymmetric(a *tensor.Tensor, n int) error {
	// The predicate reads each off-diagonal pair FOUR times, so the naive walk costs 2n²
	// interface dispatches on every CholSolve — 524k at n=512, before any arithmetic happens
	// (PS1005). The contiguous path reads the same two values once each from the typed backing
	// slice; the AtF64 walk stays as the fallback for non-f64 or strided inputs. Identical
	// comparison, identical tolerance, identical first-offending pair reported.
	if af, ok := flatRowMajor(a); ok {
		for i := range n {
			for j := i + 1; j < n; j++ {
				x, y := af[i*n+j], af[j*n+i]
				if math.Abs(x-y) > 1e-9*(math.Abs(x)+math.Abs(y))+1e-12 {
					return fmt.Errorf("linalg: expected a symmetric matrix; A[%d,%d]=%g != A[%d,%d]=%g",
						i, j, x, j, i, y)
				}
			}
		}
		return nil
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			if math.Abs(a.AtF64(i, j)-a.AtF64(j, i)) > 1e-9*(math.Abs(a.AtF64(i, j))+math.Abs(a.AtF64(j, i)))+1e-12 {
				return fmt.Errorf("linalg: expected a symmetric matrix; A[%d,%d]=%g != A[%d,%d]=%g",
					i, j, a.AtF64(i, j), j, i, a.AtF64(j, i))
			}
		}
	}
	return nil
}
