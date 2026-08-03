package linalg

import (
	"fmt"
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
	lf, err := cholFactor(a, n)
	if err != nil {
		return nil, err
	}
	// lf is already the lower-triangular L in row-major order: cholFactor writes
	// only the diagonal and sub-diagonal, leaving the strictly-upper entries at
	// their make() zero, so it IS the returned factor (no lower-triangle copy).
	return tensor.FromFloat64(tensor.Shape{n, n}, lf), nil
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
	lf, err := cholFactor(a, n)
	if err != nil {
		return nil, err
	}
	vec := b.Ndim() == 1
	cols := 1
	if !vec {
		cols = b.Shape()[1]
	}
	out := make([]float64, n*cols) // [n,cols] row-major
	// One forward-substitution buffer for the whole solve, not one per right-hand side. Nothing
	// carries across columns — every entry is written before it is read — so a fresh allocation
	// per column bought only garbage: 128 columns cost 128 allocations of it.
	// Read the right-hand side through its storage when it is already F64. AtF64 walks the
	// shape to a flat offset and dispatches on the storage type for EVERY element, and this
	// loop touches n*cols of them: it was 9.9%% of the profile at n=256, cols=8. Values are
	// identical — the same float64 out of the same storage — and any other dtype or layout
	// falls through to AtF64 unchanged.
	var bf []float64
	if bc := b.Contiguous(); bc.Dtype() == tensor.F64 {
		bf = bc.Storage().F64()
	}
	// Right-hand sides are independent: column c reads the factor and its own column of b, and
	// writes only out[·*cols+c]. Every column costs the same 2*n*n/2 substitution terms, so equal
	// bands need no balancing, and the work gate keeps a narrow solve serial. The buffer is one
	// per WORKER — hoisted out of the column loop but inside the band — so the fan-out does not
	// give back the allocation this loop just stopped making.
	parallelCols(cols, n*n, func(lo, hi int) {
		y := make([]float64, n)
		for c := lo; c < hi; c++ {
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
				li := lf[i*n : i*n+n] // contiguous factor row i (was l[i], a scattered slice)
				for k := range i {
					s -= li[k] * y[k]
				}
				y[i] = s / li[i]
			}
			for i := n - 1; i >= 0; i-- { // back: Lᵀ·x = y (Lᵀ[i,k] = L[k,i])
				s := y[i]
				for k := i + 1; k < n; k++ {
					s -= lf[k*n+i] * out[k*cols+c] // L[k,i], a strided column read (unchanged)
				}
				out[i*cols+c] = s / lf[i*n+i]
			}
		}
	})
	if vec {
		return tensor.FromFloat64(tensor.Shape{n}, out), nil
	}
	return tensor.FromFloat64(tensor.Shape{n, cols}, out), nil
}

// cholFactor computes the lower-triangular Cholesky factor L of the n×n SPD matrix a as a FLAT
// row-major []float64 (L[i][j] at lf[i*n+j]), erroring if a is not positive definite. The flat
// buffer replaces the previous [][]float64 (one heap slice per row, scattered in memory with a
// slice-header dereference on every l[i][k] access): a single contiguous block gives the O(n³)
// inner dot-product sequential, pointer-chase-free reads of the already-computed columns. The
// inner Σ l[i][k]·l[j][k] runs through dot4 — a 4-accumulator sum that breaks the single-
// accumulator dependency chain (each fma waited a full add-latency on the previous), so four
// independent chains retire in parallel. This REASSOCIATES the sum (four partial sums combined
// (s0+s1)+(s2+s3) then the <4 tail), which relaxes L off bit-identity to the incumbent tolerance
// (numpy/LAPACK dpotrf blocks the same accumulation); the SPD reconstruction A≈L·Lᵀ stays well
// inside the §V16 1e-9 gate. Same `s / L[j][j]` DIVISION (no hoisted reciprocal).
func cholFactor(a *tensor.Tensor, n int) ([]float64, error) {
	lf := make([]float64, n*n)
	// Read the input through its storage when it is already F64. AtF64 walks the shape to a flat
	// offset and dispatches on the storage type for EVERY element of the column sweep, and that
	// was 57%% of this factorization's profile at n=512 — more than four times the arithmetic it
	// guards. Values are identical; any other dtype or layout keeps the accessor.
	var af []float64
	if ac := a.Contiguous(); ac.Dtype() == tensor.F64 {
		af = ac.Storage().F64()
	}
	for j := range n {
		lj := lf[j*n : j*n+n] // contiguous row j
		ajj := a.AtF64(j, j)
		if af != nil {
			ajj = af[j*n+j]
		}
		d := ajj - dot4(lj, lj, j)
		if d <= 0 {
			return nil, fmt.Errorf("linalg: matrix is not positive definite (non-positive pivot %g at leading minor %d)", d, j)
		}
		ljj := math.Sqrt(d)
		lj[j] = ljj
		// The two arms are duplicated rather than sharing a helper: this is the inner loop of an
		// O(n^3) factorization, and routing it through a closure or a per-element branch gives
		// back what the typed read wins. Any edit has to be made twice.
		if af != nil {
			i := j + 1
			// FOUR ROWS PER PASS. Every row's dot streams the same pivot row lj; taking four at
			// once reads it once for all four and puts sixteen independent chains in flight.
			// BIT-IDENTICAL — the jammed dimension is the FREE one, and each row keeps its own
			// four accumulators over the same ascending k (see dot4x4).
			// TWO ROWS PER PASS. Every row's dot streams the SAME lj; taking two at once loads
			// it once for both and doubles the independent chains in flight. BIT-IDENTICAL:
			// each row keeps dot4's own four accumulators over the same ascending k and the
			// same ((s0+s1)+(s2+s3)) combination, so the jammed dimension is the free one.
			for ; i+3 < n; i += 4 {
				l0 := lf[(i+0)*n : (i+0)*n+j]
				l1 := lf[(i+1)*n : (i+1)*n+j]
				l2 := lf[(i+2)*n : (i+2)*n+j]
				l3 := lf[(i+3)*n : (i+3)*n+j]
				d0, d1, d2, d3 := dot4x4(l0, l1, l2, l3, lj, j)
				lf[(i+0)*n+j] = (af[(i+0)*n+j] - d0) / ljj
				lf[(i+1)*n+j] = (af[(i+1)*n+j] - d1) / ljj
				lf[(i+2)*n+j] = (af[(i+2)*n+j] - d2) / ljj
				lf[(i+3)*n+j] = (af[(i+3)*n+j] - d3) / ljj
			}
			for ; i < n; i++ {
				li := lf[i*n : i*n+n] // contiguous row i
				s := af[i*n+j] - dot4(li, lj, j)
				li[j] = s / ljj
			}
			continue
		}
		i := j + 1
		for ; i+3 < n; i += 4 { // the same jam; the two arms must not diverge
			d0, d1, d2, d3 := dot4x4(lf[(i+0)*n:(i+0)*n+j], lf[(i+1)*n:(i+1)*n+j],
				lf[(i+2)*n:(i+2)*n+j], lf[(i+3)*n:(i+3)*n+j], lj, j)
			lf[(i+0)*n+j] = (a.AtF64(i+0, j) - d0) / ljj
			lf[(i+1)*n+j] = (a.AtF64(i+1, j) - d1) / ljj
			lf[(i+2)*n+j] = (a.AtF64(i+2, j) - d2) / ljj
			lf[(i+3)*n+j] = (a.AtF64(i+3, j) - d3) / ljj
		}
		for ; i < n; i++ {
			li := lf[i*n : i*n+n]
			s := a.AtF64(i, j) - dot4(li, lj, j)
			li[j] = s / ljj
		}
	}
	return lf, nil
}

// dot4 returns Σ_{k<n} x[k]·y[k] with four independent accumulators. The single-accumulator
// form is latency-bound — every add depends on the previous, so the O(n³) factorization stalls
// a full FP-add latency per element; four chains keep the FMA units busy and combine as
// (s0+s1)+(s2+s3) plus the <4 remainder. Reassociated (not bit-identical to the ascending-k sum),
// riding the incumbent numeric tolerance.
func dot4(x, y []float64, n int) float64 {
	var s0, s1, s2, s3 float64
	k := 0
	for ; k+4 <= n; k += 4 {
		s0 += x[k] * y[k]
		s1 += x[k+1] * y[k+1]
		s2 += x[k+2] * y[k+2]
		s3 += x[k+3] * y[k+3]
	}
	s := (s0 + s1) + (s2 + s3)
	for ; k < n; k++ {
		s += x[k] * y[k]
	}
	return s
}

// dot4x4 returns the four dots Σ xr[k]·y[k] over ONE shared pass of y. Each is computed exactly
// as dot4 computes it — four accumulators over the same ascending k, combined as (s0+s1)+(s2+s3)
// plus the <4 remainder — so four jammed rows produce the factor four dot4 calls produced, bit
// for bit. What it buys is loads and chains: the pivot row is read once for four rows instead of
// four times, and sixteen independent accumulators keep the FP pipes busy where four leave them
// waiting on latency. MEASURED: Cholesky512 9.07 to 6.19 ms, -31.8%%. Jamming two rows instead of
// four gets only half of it (6.99 ms), and four rows written as two dot4x2 calls gets nothing —
// the sharing is in the SINGLE pass over y, not in the unrolling.
func dot4x4(x0, x1, x2, x3, y []float64, n int) (float64, float64, float64, float64) {
	var a0, a1, a2, a3, b0, b1, b2, b3 float64
	var c0, c1, c2, c3, e0, e1, e2, e3 float64
	k := 0
	for ; k+4 <= n; k += 4 {
		y0, y1, y2, y3 := y[k], y[k+1], y[k+2], y[k+3]
		a0 += x0[k] * y0
		a1 += x0[k+1] * y1
		a2 += x0[k+2] * y2
		a3 += x0[k+3] * y3
		b0 += x1[k] * y0
		b1 += x1[k+1] * y1
		b2 += x1[k+2] * y2
		b3 += x1[k+3] * y3
		c0 += x2[k] * y0
		c1 += x2[k+1] * y1
		c2 += x2[k+2] * y2
		c3 += x2[k+3] * y3
		e0 += x3[k] * y0
		e1 += x3[k+1] * y1
		e2 += x3[k+2] * y2
		e3 += x3[k+3] * y3
	}
	sa, sb := (a0+a1)+(a2+a3), (b0+b1)+(b2+b3)
	sc, se := (c0+c1)+(c2+c3), (e0+e1)+(e2+e3)
	for ; k < n; k++ {
		sa += x0[k] * y[k]
		sb += x1[k] * y[k]
		sc += x2[k] * y[k]
		se += x3[k] * y[k]
	}
	return sa, sb, sc, se
}

// requireSymmetric errors if a is not symmetric to a relative tolerance.
func requireSymmetric(a *tensor.Tensor, n int) error {
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
