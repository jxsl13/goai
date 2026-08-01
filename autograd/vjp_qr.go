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
		for i := range n {
			mm[i] = make([]float64, n)
			rdi, mmi := rd[i], mm[i]
			for j := range n {
				var s float64
				rbj := rb[j]
				for k := range n { // (R·R̄ᵀ)_ij = Σ_k R[i,k]·R̄[j,k]
					s += rdi[k] * rbj[k]
				}
				mmi[j] = s
			}
		}
		for k := range m { // −(Q̄ᵀ·Q)_ij = −Σ_k Q̄[k,i]·Q[k,j]
			qbk, qdk := qb[k], qd[k]
			for i := range n {
				qbki, mmi := qbk[i], mm[i]
				for j := range n {
					mmi[j] -= qbki * qdk[j]
				}
			}
		}
		// copyltu(M): symmetric from the lower triangle (diagonal once).
		c := make([][]float64, n)
		for i := range n {
			c[i] = make([]float64, n)
		}
		for i := range n {
			for j := range n {
				if i >= j {
					c[i][j] = mm[i][j]
				} else {
					c[i][j] = mm[j][i] // mirror strict-lower into the upper
				}
			}
		}
		// B = Q̄ + Q·copyltu(M)  (m×n). Same interchange: c is read down a column with j innermost,
		// so k moves outside and c[k] becomes a contiguous row. Each B[i][j] still starts from
		// Q̄[i][j] and adds the n terms in ascending k.
		b := make([][]float64, m)
		for i := range m {
			bi := make([]float64, n)
			b[i] = bi
			copy(bi, qb[i])
			qdi := qd[i]
			for k := range n {
				qdik, ck := qdi[k], c[k]
				for j := range n {
					bi[j] += qdik * ck[j]
				}
			}
		}
		// Rinv = R⁻¹ (upper-triangular) by back-substitution on R·X = I.
		rinv := make([][]float64, n)
		for i := range n {
			rinv[i] = make([]float64, n)
		}
		for col := range n {
			rinv[col][col] = 1 / rd[col][col]
			for i := col - 1; i >= 0; i-- {
				var s float64
				for k := i + 1; k <= col; k++ {
					s += rd[i][k] * rinv[k][col]
				}
				rinv[i][col] = -s / rd[i][i]
			}
		}
		// Ā = B·R⁻ᵀ : Ā[i,j] = Σ_k B[i,k]·(R⁻ᵀ)[k,j] = Σ_k B[i,k]·Rinv[j,k].
		abar := tensor.New(q.Dtype(), tensor.Shape{m, n})
		for i := range m {
			for j := range n {
				var s float64
				for k := j; k < n; k++ { // Rinv upper-tri ⇒ Rinv[j,k] nonzero for k ≥ j
					s += b[i][k] * rinv[j][k]
				}
				abar.SetF64(s, i, j)
			}
		}
		return []*tensor.Tensor{abar}, nil
	})
}

// to2D copies a rank-2 tensor into a fresh [rows][cols] f64 slice.
func to2D(t *tensor.Tensor, rows, cols int) [][]float64 {
	d := make([][]float64, rows)
	for i := range rows {
		d[i] = make([]float64, cols)
		for j := range cols {
			d[i][j] = t.AtF64(i, j)
		}
	}
	return d
}
