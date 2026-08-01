package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SVD VJP (Townsend 2016 "Differentiating the Singular Value Decomposition";
// Ionescu, Vantzos & Sminchisescu 2015 arXiv:1509.07838; PyTorch svd_backward). For
// the thin A = U·diag(s)·Vᵀ (m≥n, U m×n / V n×n orthonormal columns, s descending)
// with cotangents Ū, s̄, V̄:
//
//	F_ij = 1/(s_j² − s_i²) for i≠j, 0 on the diagonal
//	J = F ∘ (UᵀŪ − ŪᵀU),   K = F ∘ (VᵀV̄ − V̄ᵀV)
//	Ā = U·[diag(s̄) + J·diag(s) + diag(s)·K]·Vᵀ            (square part)
//	    + (I − U·Uᵀ)·Ū·diag(1/s)·Vᵀ                       (m>n only)
//
// The antisymmetric (skew) J,K construction makes the U/V gradient invariant to the
// per-column sign gauge. The singular-value term U·diag(s̄)·Vᵀ is exact and
// degeneracy-free; the U/V terms need DISTINCT, nonzero singular values (F blows up
// at a repeated singular value, 1/s at a zero one — documented, as in PyTorch). The
// (I−UUᵀ) term captures the part of Ū outside col(U) for a strictly tall matrix; it
// vanishes when m=n (V is square so no analogous (I−VVᵀ) term). MULTI-output VJP:
// gouts = [Ū, s̄, V̄] (zero where unused). All arithmetic is f64 (§V10).
func init() {
	RegisterVJPMulti(backend.OpSVD, func(_ *backend.Context, _, outputs []*tensor.Tensor, _ backend.Attrs, gouts []*tensor.Tensor) ([]*tensor.Tensor, error) {
		ut, st, vt := outputs[0], outputs[1], outputs[2]
		ubar, sbar, vbar := gouts[0], gouts[1], gouts[2]
		m := ut.Shape()[0]
		n := st.Shape()[0]

		s := make([]float64, n)
		sb := make([]float64, n)
		for i := range n {
			s[i] = st.AtF64(i)
			sb[i] = sbar.AtF64(i)
		}
		u := to2D(ut, m, n)
		v := to2D(vt, n, n)
		ub := to2D(ubar, m, n)
		vb := to2D(vbar, n, n)

		// skewU = UᵀŪ − ŪᵀU, skewV = VᵀV̄ − V̄ᵀV (n×n).
		utu := matTmulRect(u, ub, m, n) // Uᵀ·Ū
		vtv := matTmulRect(v, vb, n, n) // Vᵀ·V̄
		// mid = diag(s̄) + J·diag(s) + diag(s)·K, J=F∘skewU, K=F∘skewV.
		mid := make([][]float64, n)
		for i := range n {
			mid[i] = make([]float64, n)
			for j := range n {
				if i == j {
					mid[i][j] = sb[i]
					continue
				}
				f := 1.0 / (s[j]*s[j] - s[i]*s[i]) // F_ij
				jU := f * (utu[i][j] - utu[j][i])  // J_ij
				kV := f * (vtv[i][j] - vtv[j][i])  // K_ij
				mid[i][j] = jU*s[j] + s[i]*kV      // (J·diag(s) + diag(s)·K)_ij
			}
		}
		// Ā_core = U·mid·Vᵀ : T = mid·Vᵀ, then U·T.
		tmp := make([][]float64, n)
		for a := range n {
			tmp[a] = make([]float64, n)
			for j := range n {
				var sum float64 // (mid·Vᵀ)_aj = Σ_b mid[a,b]·V[j,b]
				for b := range n {
					sum += mid[a][b] * v[j][b]
				}
				tmp[a][j] = sum
			}
		}
		abar := tensor.New(ut.Dtype(), tensor.Shape{m, n})
		for i := range m {
			for j := range n {
				var sum float64 // (U·T)_ij = Σ_a U[i,a]·T[a,j]
				for a := range n {
					sum += u[i][a] * tmp[a][j]
				}
				abar.SetF64(sum, i, j)
			}
		}

		// tall correction (m>n): (I − U·Uᵀ)·Ū·diag(1/s)·Vᵀ, added to Ā.
		if m > n {
			w := make([][]float64, m) // W = Ū·diag(1/s)
			for i := range m {
				w[i] = make([]float64, n)
				for j := range n {
					w[i][j] = ub[i][j] / s[j]
				}
			}
			utw := matTmulRect(u, w, m, n) // Uᵀ·W (n×n)
			addTallCorrection(abar, w, u, utw, v, m, n)
		}
		return []*tensor.Tensor{abar}, nil
	})
}

// addTallCorrection adds (proj·Vᵀ) to Ā, where proj = W − U·(Uᵀ·W) is the part of Ū outside col(U)
// — the term that exists only for a strictly tall matrix.
//
// The projection row depends on i and b ONLY. The original rebuilt it INSIDE the j loop, n times
// over, which made this term O(m·n³) where O(m·n²) suffices; hoisting it deletes an entire factor
// of n of arithmetic rather than merely improving locality.
//
// It is a named function rather than an inline loop so it can be gated: a test comparing two
// implementations it defines itself proves a mathematical identity but cannot detect a change to
// the shipped code, which is what the first version of that test did.
//
// BIT-IDENTICAL: each proj[b] is the same subtraction sequence over ascending a, and each add still
// accumulates over ascending b with the same operands.
func addTallCorrection(abar *tensor.Tensor, w, u, utw, v [][]float64, m, n int) {
	proj := make([]float64, n)
	for i := range m {
		wi, ui := w[i], u[i]
		for b := range n {
			proj[b] = wi[b]
		}
		for a := range n {
			uia, utwa := ui[a], utw[a]
			for b := range n {
				proj[b] -= uia * utwa[b]
			}
		}
		for j := range n {
			var add float64
			vj := v[j]
			for b := range n {
				add += proj[b] * vj[b]
			}
			abar.SetF64(abar.AtF64(i, j)+add, i, j)
		}
	}
}

// matTmulRect returns Xᵀ·Y where X is p×q and Y is p×q, giving a q×q matrix
// (Xᵀ·Y)[i,j] = Σ_k X[k,i]·Y[k,j]. It computes it for two p-by-q matrices, with the CONTRACTION index outermost.
//
// Written the obvious way — s accumulating over k innermost — both operands are read down a column
// and every step jumps a whole row to use eight bytes. The k loop only accumulates, so it can move
// outside; then x[k] and y[k] are contiguous rows and out[i] is a row (§INTERCHANGE-BEFORE-TRANSPOSE,
// where the same move beat transposing three to one at zero memory cost).
//
// BIT-IDENTICAL: every out[i][j] still sums the same p products in ascending k order.
func matTmulRect(x, y [][]float64, p, q int) [][]float64 {
	out := make([][]float64, q)
	for i := range q {
		out[i] = make([]float64, q)
	}
	for k := range p {
		xk, yk := x[k], y[k]
		for i := range q {
			xki, outi := xk[i], out[i]
			for j := range q {
				outi[j] += xki * yk[j]
			}
		}
	}
	return out
}
