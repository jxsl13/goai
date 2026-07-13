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
			for i := range m {
				for j := range n {
					// proj = (W − U·(Uᵀ·W)); add (proj·Vᵀ)_ij.
					var add float64
					for b := range n {
						projib := w[i][b]
						for a := range n {
							projib -= u[i][a] * utw[a][b]
						}
						add += projib * v[j][b]
					}
					abar.SetF64(abar.AtF64(i, j)+add, i, j)
				}
			}
		}
		return []*tensor.Tensor{abar}, nil
	})
}

// matTmulRect returns Xᵀ·Y where X is p×q and Y is p×q, giving a q×q matrix
// (Xᵀ·Y)[i,j] = Σ_k X[k,i]·Y[k,j].
func matTmulRect(x, y [][]float64, p, q int) [][]float64 {
	out := make([][]float64, q)
	for i := range q {
		out[i] = make([]float64, q)
		for j := range q {
			var s float64
			for k := range p {
				s += x[k][i] * y[k][j]
			}
			out[i][j] = s
		}
	}
	return out
}
