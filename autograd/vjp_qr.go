package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// QR VJP (Seeger, Hetzel, Dai, Meissner & Lawrence 2017 "Auto-Differentiating
// Linear Algebra", arXiv:1710.08717; Townsend; PyTorch/TF linalg_qr_backward). For
// the reduced A = Q·R (m≥n, Q m×n orthonormal cols, R n×n upper-tri) with output
// cotangents Q̄, R̄:
//
//	M = R·R̄ᵀ − Q̄ᵀ·Q               (n×n)
//	Ā = [Q̄ + Q·copyltu(M)]·R⁻ᵀ
//
// where copyltu(X) = tril(X) + tril(X,−1)ᵀ is the symmetric matrix built from X's
// lower triangle (diagonal taken once). The full (unprojected) Q̄ term already
// carries the m>n orthogonal-complement contribution (I−QQᵀ)Q̄·R⁻ᵀ, so one formula
// covers m=n and m>n. This is a MULTI-output VJP: gouts = [Q̄, R̄] (a zero tensor
// where an output is unused). All arithmetic is f64 (§V10).
func init() {
	RegisterVJPMulti(backend.OpQR, func(_ *backend.Context, _, outputs []*tensor.Tensor, _ backend.Attrs, gouts []*tensor.Tensor) ([]*tensor.Tensor, error) {
		q, r := outputs[0], outputs[1]
		qbar, rbar := gouts[0], gouts[1]
		m := q.Shape()[0]
		n := q.Shape()[1]

		// dense f64 copies
		qd := to2D(q, m, n)
		rd := to2D(r, n, n)
		qb := to2D(qbar, m, n)
		rb := to2D(rbar, n, n)

		// M = R·R̄ᵀ − Q̄ᵀ·Q  (n×n)
		//
		// The Q̄ᵀ·Q term is the O(n²m) one and both its operands are read DOWN a column. Rather
		// than transpose them — which was tried and cost 38%% more bytes for 12%% less time — the
		// k loop is moved OUTSIDE: for a fixed k, Q̄[k] and Q[k] are contiguous rows and M[i] is a
		// row, so every access streams and nothing extra is allocated.
		//
		// BIT-IDENTICAL: each M[i][j] still takes the R·R̄ᵀ sum first and then subtracts the m
		// terms in ascending k, exactly as before. Only the loop nesting changed.
		mm := make([][]float64, n)
		//perfscan:ignore PS3034,PS3043 stale line; not flagged by current perfscan on optimized file | stale line; part of M-dot, covered by PS4008 f
		for i := range n {
			//perfscan:ignore PS2008,PS3064 resource-only alloc class, no wall-clock win | stale line; not flagged by current perfscan
			mm[i] = make([]float64, n)
			rdi, mmi := rd[i], mm[i]
			//perfscan:ignore PS6010 resource-only invariant class, no wall-clock win
			for j := range n {
				var s float64
				rbj := rb[j]
				//perfscan:ignore PS3010 stale line; live construct covered by PS4008 flag
				for k := range n { // (R·R̄ᵀ)_ij = Σ_k R[i,k]·R̄[j,k]
					s += rdi[k] * rbj[k]
				}
				mmi[j] = s
			}
		}
		// EIGHT ROWS OF Q PER PASS. M does not vary with k, so the loop above walked the whole
		// n x n matrix once per row of Q — a load and a store of M[i][j] for one subtraction
		// each, which was 50.7% of this benchmark on a single line. Taking eight k at once
		// loads and stores M[i][j] once for eight subtractions. Eight is the measured
		// optimum: widths 12 and 16 regress on register pressure (8.38 and 8.35 ms against
		// 7.87), and every width from 2 to 16 leaves the digests unchanged.
		//
		// BIT-IDENTICAL: each M[i][j] still sees k ascending, and the accumulator is an
		// EXPLICIT LOCAL rather than a compound assignment — `mmi[j] -= a0*x0 + a1*x1 + ...`
		// would subtract the SUM of the terms, which associates differently (T1183).
		k := 0
		//perfscan:ignore PS3066 stale line; B loop is the PS4006 flatten target, covered
		for ; k+7 < m; k += 8 {
			qb0, qb1, qb2, qb3, qb4, qb5, qb6, qb7 := qb[k+0], qb[k+1], qb[k+2], qb[k+3], qb[k+4], qb[k+5], qb[k+6], qb[k+7]
			qd0, qd1, qd2, qd3, qd4, qd5, qd6, qd7 := qd[k+0], qd[k+1], qd[k+2], qd[k+3], qd[k+4], qd[k+5], qd[k+6], qd[k+7]
			for i := range n {
				a0, a1, a2, a3, a4, a5, a6, a7 := qb0[i], qb1[i], qb2[i], qb3[i], qb4[i], qb5[i], qb6[i], qb7[i]
				mmi := mm[i]
				for j := range n {
					v := mmi[j]
					v -= a0 * qd0[j]
					v -= a1 * qd1[j]
					v -= a2 * qd2[j]
					v -= a3 * qd3[j]
					v -= a4 * qd4[j]
					v -= a5 * qd5[j]
					v -= a6 * qd6[j]
					v -= a7 * qd7[j]
					mmi[j] = v
				}
			}
		}
		for ; k < m; k++ {
			qbk, qdk := qb[k], qd[k]
			for i := range n {
				qbki, mmi := qbk[i], mm[i]
				for j := range n {
					mmi[j] -= qbki * qdk[j]
				}
			}
		}
		// copyltu(M): symmetric from the lower triangle (diagonal once).
		c := alloc2D(n, n)
		for i := range n {
			for j := range n {
				if i >= j {
					//perfscan:ignore PS3016 matmul-class covered by PS4006 flatten flag
					c[i][j] = mm[i][j]
				} else {
					//perfscan:ignore PS3016 matmul-class covered by PS4006 flatten flag
					c[i][j] = mm[j][i] // mirror strict-lower into the upper
				}
			}
		}
		// B = Q̄ + Q·copyltu(M)  (m×n). Same interchange: c is read down a column with j innermost,
		// so k moves outside and c[k] becomes a contiguous row. Each B[i][j] still starts from
		// Q̄[i][j] and adds the n terms in ascending k.
		//
		// Rows are independent — row i reads only qb[i], qd[i] and all of c, and writes only
		// b[i] — and the per-row cost is the same n*n for every row, so a striped split needs no
		// balancing. Every B[i][j] still adds its n terms in ascending k, so the result is
		// bit-identical to the serial loop and the parity test asserts exact equality.
		b := make([][]float64, m)
		logdetParallelIdx(m, m*n*n, func(i int) {
			//perfscan:ignore PS6008 resource-only class, no wall-clock win
			bi := make([]float64, n)
			b[i] = bi
			copy(bi, qb[i])
			qdi := qd[i]
			for k := range n {
				qdik, ck := qdi[k], c[k]
				for j := range n {
					//perfscan:ignore PS3075 stale line; not flagged by current perfscan
					bi[j] += qdik * ck[j]
				}
			}
		})
		// Rinv = R⁻¹ (upper-triangular) by back-substitution on R·X = I.
		rinv := alloc2D(n, n)
		//perfscan:ignore PS3063 stale line; not flagged by current perfscan
		for col := range n {
			rinv[col][col] = 1 / rd[col][col]
			//perfscan:ignore PS3040 stale line; not flagged by current perfscan
			for i := col - 1; i >= 0; i-- {
				var s float64
				//perfscan:ignore PS3010,PS4012 stale line; not flagged by current perfscan | stale line; scaled-serial-dot not present in optimized file
				for k := i + 1; k <= col; k++ {
					//perfscan:ignore PS1010,PS3016 stale line; not flagged by current perfscan | matmul-class covered by PS4006 flatten flag
					s += rd[i][k] * rinv[k][col]
				}
				//perfscan:ignore PS3016 matmul-class covered by PS4006 flatten flag
				rinv[i][col] = -s / rd[i][i]
			}
		}
		// Ā = B·R⁻ᵀ : Ā[i,j] = Σ_k B[i,k]·(R⁻ᵀ)[k,j] = Σ_k B[i,k]·Rinv[j,k].
		abar := tensor.New(q.Dtype(), tensor.Shape{m, n})
		for i := range m {
			for j := range n {
				var s float64
				//perfscan:ignore PS3010 stale line; live construct covered by PS4006 flag
				for k := j; k < n; k++ { // Rinv upper-tri ⇒ Rinv[j,k] nonzero for k ≥ j
					//perfscan:ignore PS3016 matmul-class covered by PS4006 flatten flag
					s += b[i][k] * rinv[j][k]
				}
				abar.SetF64(s, i, j)
			}
		}
		return []*tensor.Tensor{abar}, nil
	})
}

// alloc2D returns a [rows][cols] f64 matrix whose rows are windows on ONE backing block.
// Callers see the same [][]float64 they always did — no call site changes — but the allocation
// count drops from rows+1 to 2 and the rows become contiguous with each other, which is what
// the LU factorization's own comment records as the reason it stopped using a jagged matrix.
// Rows must not be appended to; nothing here does, and a row window is capped at its own length
// so an append would copy rather than reach into its neighbor.
//
//perfscan:ignore PS3033 stale line; not flagged by current perfscan
func alloc2D(rows, cols int) [][]float64 {
	base := make([]float64, rows*cols)
	d := make([][]float64, rows)
	for i := range rows {
		d[i] = base[i*cols : (i+1)*cols : (i+1)*cols]
	}
	return d
}

// to2D copies a rank-2 tensor into a fresh [rows][cols] f64 matrix.
//
// The per-element AtF64 walk is kept only as the fallback. For a contiguous, offset-0 tensor
// the storage is read through its typed slice instead: exact for F64, and an exact widening for
// F32, so every value is the one AtF64 would have returned. At the QR adjoint's shape that walk
// was 1452 allocations per call between this and the matrices below it.
func to2D(t *tensor.Tensor, rows, cols int) [][]float64 {
	d := alloc2D(rows, cols)
	if t.IsContiguous() && t.Offset() == 0 {
		switch t.Dtype() {
		case tensor.F64:
			ts := t.Storage().F64()
			for i := range rows {
				copy(d[i], ts[i*cols:(i+1)*cols])
			}
			return d
		case tensor.F32:
			ts := t.Storage().F32()
			for i := range rows {
				row := d[i]
				for j, v := range ts[i*cols : (i+1)*cols] {
					row[j] = float64(v) // exact widening, as AtF64 does
				}
			}
			return d
		}
	}
	for i := range rows {
		row := d[i]
		for j := range cols {
			row[j] = t.AtF64(i, j)
		}
	}
	return d
}
