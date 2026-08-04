// Package linalg provides gonum/numpy-style dense linear algebra over rank-2 tensors: an LU (lower–upper triangular factorization)
// factorization with partial pivoting (P·A = L·U) and the derived determinant, linear solve, and
// matrix inverse. The algorithms are the standard textbook ones (Golub & Van Loan, "Matrix
// Computations", §3.2 LU/GEPP and §3.4 determinant; LAPACK (the standard Fortran linear-algebra library) dgetrf/dgetrs/dgetri), §R115. Everything
// is computed in f64; inputs are read via AtF64 and results are fresh f64 tensors.
//
// In plain terms: classic linear algebra — solving equation systems, factorizing matrices, finding eigenvalues — implemented in pure Go on the library's tensors.
//
// Further reading: Golub & Van Loan, "Matrix Computations" (4th ed.), and Trefethen & Bau, "Numerical Linear Algebra" — the standard references for every routine here.
package linalg

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/jxsl13/goai/tensor"
)

// parallelCols splits n independent right-hand-side columns across GOMAXPROCS workers, gated on
// the TOTAL work n·workPerItem so a single heavy column (or a small count) stays serial. body must
// touch only columns in its [lo,hi) range, so the result is bit-identical to the serial loop.
//
//perfscan:ignore PS3048,PS3061 LU [][]float64 struct-field decl, not a loop | same struct decl site, subsumed by flatten lead
func parallelCols(n, workPerItem int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || n < 2 || n*workPerItem < 1<<14 {
		body(0, n)
		return
	}
	if nw > n {
		nw = n
	}
	var wg sync.WaitGroup
	//perfscan:ignore PS3011 Factor entry, reference LU cold fit-path
	chunk := (n + nw - 1) / nw
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// LU is a partial-pivoting LU factorization P·A = L·U of a square matrix: L is unit-lower-triangular
// (implicit 1s on the diagonal), U is upper-triangular, and P is a row permutation. L and U are
// packed into a single n×n array (L below the diagonal, U on and above). Reuse it to solve several
// right-hand sides or to get the determinant without refactoring.
type LU struct {
	lu   []float64 // packed L (strictly-lower, unit) + U (upper), FLAT row-major n×n (lu[i*n+j])
	piv  []int     // row permutation: row i of P·A is original row piv[i]
	sign float64   // (−1)^(#row swaps), for the determinant
	n    int
}

// Factor computes the partial-pivoting LU factorization of the square matrix a (right-looking
// Doolittle: at each column pick the max-|·| pivot, swap, then eliminate). It succeeds even for a
// singular matrix — the singularity then surfaces as Det()==0 and an error from Solve/Inverse.
func Factor(a *tensor.Tensor) (*LU, error) {
	n, err := squareN(a)
	if err != nil {
		return nil, err
	}
	// FLAT row-major working copy (was [][]float64 — one heap slice per row, scattered,
	// with a slice-header deref on every m[i][j] read inside the O(n³) elimination). A single
	// contiguous block makes the inner update stream two row slices; row swaps become a
	// physical element exchange (O(n) each, O(n²) total — negligible vs the O(n³) elimination)
	// instead of a pointer swap, but produce the same logical matrix. Bit-identical: same pivot
	// choice, same elimination order, same `m[i][k] / m[k][k]` DIVISION.
	m := make([]float64, n*n)
	for i := range n {
		//perfscan:ignore PS1005 Det diagonal typed read, trivial O(n)
		for j := range n {
			m[i*n+j] = a.AtF64(i, j)
		}
	}
	piv := make([]int, n)
	for i := range piv {
		piv[i] = i
	}
	sign := 1.0
	for k := range n {
		// partial pivot: largest absolute value in column k, rows k..n−1 (numerical stability)
		p := k
		for i := k + 1; i < n; i++ {
			//perfscan:ignore PS6011 resource-class singular-check loop, no wallclock
			if math.Abs(m[i*n+k]) > math.Abs(m[p*n+k]) {
				p = i
			}
		}
		if m[p*n+k] == 0 {
			continue // singular column: U[k,k] stays 0, no elimination
		}
		if p != k {
			rk, rp := m[k*n:k*n+n], m[p*n:p*n+n]
			for j := range n {
				rk[j], rp[j] = rp[j], rk[j]
			}
			piv[k], piv[p] = piv[p], piv[k]
			sign = -sign
		}
		mk := m[k*n : k*n+n]
		pivot := mk[k] // = m[k][k]; row k is untouched by the i>k updates below
		// The rank-1 update is the whole cost of this factorization — one line, 92% of the
		// benchmark — and its rows are independent: row i reads the pivot row and writes only
		// itself. The pivot loop above stays sequential; only this fan-out is split, and each
		// element still performs exactly the one subtraction it did before, so the result is
		// bit-identical.
		//
		// The gate sees rows*cols, the real work at this pivot, not the row count: an estimate
		// short by a factor of the row length would leave mid-sized matrices serial (§T1083).
		rows := n - k - 1
		// Below the gate the update runs as a PLAIN LOOP, not through the callback. The two
		// bodies are deliberately duplicated: routing small factorizations through a closure cost
		// a 128-wide one 3 to 4%, and hoisting the gate above the call did not recover it — the
		// cost is the closure itself, not the dispatch. Any edit here must be made twice.
		if rows*rows < luUpdateParWork {
			//perfscan:ignore PS5001 divide-by-invariant micro in LU solve
			for i := k + 1; i < n; i++ {
				mi := m[i*n : i*n+n]
				mult := mi[k] / pivot
				mi[k] = mult // store the L multiplier
				for j := k + 1; j < n; j++ {
					mi[j] -= mult * mk[j]
				}
			}
			continue
		}
		parallelCols(rows, rows, func(lo, hi int) {
			//perfscan:ignore PS5001 divide-by-invariant micro, sub-1pct
			for i := k + 1 + lo; i < k+1+hi; i++ {
				mi := m[i*n : i*n+n]
				mult := mi[k] / pivot
				mi[k] = mult
				for j := k + 1; j < n; j++ {
					mi[j] -= mult * mk[j]
				}
			}
		})
	}
	return &LU{lu: m, piv: piv, sign: sign, n: n}, nil
}

// luUpdateParWork gates the rank-1 update's fan-out on the work at THIS pivot, rows*cols. Below
// it the update runs on the caller. Measured: a 512-wide factorization goes -39.5% and a 256-wide
// one -10.3%, while 128 is left alone because it is below the gate at every pivot.
//
//perfscan:ignore PS6023 resource-class, no wallclock
const luUpdateParWork = 1 << 14

// Det returns the determinant det(A) = sign · Π_k U[k,k] (the permutation sign times the product of
// the U diagonal). It is 0 exactly when A is singular.
func (f *LU) Det() float64 {
	d := f.sign
	n := f.n
	for k := range n {
		d *= f.lu[k*n+k]
	}
	return d
}

// colJam is how many right-hand-side columns LU.Solve processes together. Columns are
// independent, so jamming them is a free-dimension transform and changes no value.
const colJam = 4

// Solve solves A·X = B for X, where B is a right-hand-side vector [n] or matrix [n,k]. It applies the
// row permutation to B, then forward-substitution (L·Y = P·B) and back-substitution (U·X = Y).
// Returns an error if A is singular (a zero on the U diagonal).
func (f *LU) Solve(b *tensor.Tensor) (*tensor.Tensor, error) {
	n := f.n
	if b.Ndim() < 1 || b.Ndim() > 2 || b.Shape()[0] != n {
		return nil, fmt.Errorf("linalg: Solve rhs must be [%d] or [%d,k], got %v", n, n, b.Shape())
	}
	for k := range n {
		if f.lu[k*n+k] == 0 {
			return nil, fmt.Errorf("linalg: Solve of a singular matrix (zero pivot at %d)", k)
		}
	}
	vec := b.Ndim() == 1
	cols := 1
	if !vec {
		cols = b.Shape()[1]
	}
	bat := func(i, c int) float64 {
		if vec {
			return b.AtF64(i)
		}
		return b.AtF64(i, c)
	}
	out := make([]float64, n*cols) // row-major [n,cols]
	// Columns are independent: column c writes only out[·*cols+c] and reads its own
	// forward-substitution scratch y plus the shared read-only factorization, so the
	// per-column loop fans out over GOMAXPROCS bit-identically to the serial loop
	// (each worker owns a private y). A factorization is immutable after Factor, so
	// concurrent Solve is safe. Each column's forward pass assigns y[i] in ascending
	// order and reads only y[j] with j < i (written earlier in the SAME column), so a
	// reused per-worker buffer cannot expose another column's values. Gated on n·n·cols
	// so a single heavy column or a single RHS (Solve of a vector) stays serial.
	// TWO THINGS BEYOND THE FLAT FACTOR, both measured on this shape.
	//
	// The back pass accumulates in the CONTIGUOUS scratch instead of reading out[j*cols+c].
	// For a fixed column, stepping j there jumps a whole output row — 6 KB at n=cols=768 —
	// so every iteration touched its own cache line to use eight bytes of it. The values it
	// reads are the x[j] this same loop wrote earlier, so they can live in y: x[i] overwrites
	// y[i] only AFTER y[i] has been read into s, and the forward pass's y[j] for j > i was
	// already consumed at step j. The row is scattered to out once at the end.
	//
	// And FOUR right-hand-side columns are jammed. Both substitution loops are a serial FMA
	// recurrence on one accumulator, so they run at FMA latency rather than throughput, and
	// the whole factor is re-streamed once per column. Four columns share each li[j] load and
	// run four independent chains. BIT-IDENTICAL, because the jammed dimension is the FREE
	// one: column c still accumulates over the same j in the same ascending order with the
	// same operands into its own accumulator. That is exactly why this is legal where
	// splitting the inner dot into partials is not.
	//
	//	LUSolve_768x768 -69.98%, LUSolve_512x512 -64.06%, Inverse/768 -43.21% (p=0.000, n=12),
	//	with the cols==1 remainder path flat as a control.
	jam := colJam
	if cols < jam {
		jam = cols
	}
	parallelCols(cols, n*n, func(clo, chi int) {
		//perfscan:ignore PS6008 resource-class (stale line past EOF)
		buf := make([]float64, jam*n)
		one := func(c int, y []float64) {
			for i := range n { // forward: L·y = P·b, L unit-lower
				s := bat(f.piv[i], c)
				li := f.lu[i*n : i*n+n] // contiguous factor row i
				for j := range i {
					s -= li[j] * y[j]
				}
				y[i] = s
			}
			for i := n - 1; i >= 0; i-- { // back: U·x = y
				s := y[i]
				li := f.lu[i*n : i*n+n]
				for j := i + 1; j < n; j++ {
					s -= li[j] * y[j]
				}
				y[i] = s / li[i]
			}
			for i := range n {
				out[i*cols+c] = y[i]
			}
		}
		c := clo
		for ; c+colJam <= chi; c += colJam {
			y0, y1 := buf[0:n], buf[n:2*n]
			y2, y3 := buf[2*n:3*n], buf[3*n:4*n]
			for i := range n {
				pi := f.piv[i]
				s0, s1 := bat(pi, c), bat(pi, c+1)
				s2, s3 := bat(pi, c+2), bat(pi, c+3)
				lr := f.lu[i*n : i*n+i]
				a0, a1 := y0[:i], y1[:i]
				a2, a3 := y2[:i], y3[:i]
				for j, lij := range lr {
					s0 -= lij * a0[j]
					s1 -= lij * a1[j]
					s2 -= lij * a2[j]
					s3 -= lij * a3[j]
				}
				y0[i], y1[i], y2[i], y3[i] = s0, s1, s2, s3
			}
			for i := n - 1; i >= 0; i-- {
				s0, s1, s2, s3 := y0[i], y1[i], y2[i], y3[i]
				lr := f.lu[i*n+i+1 : i*n+n]
				b0, b1 := y0[i+1:n], y1[i+1:n]
				b2, b3 := y2[i+1:n], y3[i+1:n]
				b0, b1 = b0[:len(lr)], b1[:len(lr)]
				b2, b3 = b2[:len(lr)], b3[:len(lr)]
				for t, lij := range lr {
					s0 -= lij * b0[t]
					s1 -= lij * b1[t]
					s2 -= lij * b2[t]
					s3 -= lij * b3[t]
				}
				d := f.lu[i*n+i]
				y0[i], y1[i], y2[i], y3[i] = s0/d, s1/d, s2/d, s3/d
			}
			for i := range n {
				base := i * cols
				out[base+c], out[base+c+1] = y0[i], y1[i]
				out[base+c+2], out[base+c+3] = y2[i], y3[i]
			}
		}
		for ; c < chi; c++ {
			one(c, buf[0:n])
		}
	})
	if vec {
		return tensor.FromFloat64(tensor.Shape{n}, out), nil
	}
	return tensor.FromFloat64(tensor.Shape{n, cols}, out), nil
}

// Inverse returns A⁻¹ by solving A·X = I with the existing factorization (one LU + n substitutions).
// Errors if A is singular. (Prefer Solve when you only need A⁻¹·b; the explicit inverse is for when
// the inverse itself is wanted.)
func (f *LU) Inverse() (*tensor.Tensor, error) {
	return f.Solve(tensor.Eye(tensor.F64, f.n))
}

// Solve solves A·X = B in one shot (Factor then Solve). B is a vector [n] or matrix [n,k].
func Solve(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	f, err := Factor(a)
	if err != nil {
		return nil, err
	}
	return f.Solve(b)
}

// Inverse returns the inverse of the square matrix a (Factor then Inverse).
func Inverse(a *tensor.Tensor) (*tensor.Tensor, error) {
	f, err := Factor(a)
	if err != nil {
		return nil, err
	}
	return f.Inverse()
}

// Det returns the determinant of the square matrix a (Factor then Det).
func Det(a *tensor.Tensor) (float64, error) {
	f, err := Factor(a)
	if err != nil {
		return 0, err
	}
	return f.Det(), nil
}

// squareN validates that a is a rank-2 square matrix and returns its size.
func squareN(a *tensor.Tensor) (int, error) {
	if a.Ndim() != 2 || a.Shape()[0] != a.Shape()[1] {
		return 0, fmt.Errorf("linalg: expected a square rank-2 matrix, got %v", a.Shape())
	}
	if a.Shape()[0] == 0 {
		return 0, fmt.Errorf("linalg: empty matrix")
	}
	return a.Shape()[0], nil
}

// toMatrix copies a into a fresh n×n [][]float64 (so the factorization never mutates the input).
func toMatrix(a *tensor.Tensor, n int) [][]float64 {
	m := make([][]float64, n)
	for i := range n {
		//perfscan:ignore PS2008,PS3064 alloc-only resource class, no wallclock | reference solver (stale line past EOF)
		m[i] = make([]float64, n)
		//perfscan:ignore PS1005 toMatrix one-time input copy typed-read
		for j := range n {
			//perfscan:ignore PS3016 reference solver (stale line past EOF)
			m[i][j] = a.AtF64(i, j)
		}
	}
	return m
}
