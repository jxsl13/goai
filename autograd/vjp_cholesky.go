package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Cholesky VJP — the reverse-mode Cholesky of Murray 2016 (arXiv:1602.07527,
// eq. 6/9/10). For L = chol(A) (lower, A = L·Lᵀ) with output cotangent L̄, form
//
//	P = Φ(Lᵀ·L̄),   S = L⁻ᵀ·P·L⁻¹,   Ā = ½(S + Sᵀ)
//
// where Φ(M) is the lower triangle of M with its diagonal halved (Φ_ij = M_ij for
// i>j, ½M_ii for i=j, 0 for i<j). The final ½(S+Sᵀ) symmetrization is the standard
// autograd convention (PyTorch/JAX): it makes Ā symmetric with ⟨Ā, dA⟩ = ⟨L̄, dL⟩
// under the tape's full Frobenius inner product for any symmetric perturbation dA
// (Murray's own eq. 9 form S+Sᵀ−diag(S) instead halves nothing off-diagonal and
// would double-count under full-Frobenius composition). Only the lower triangle of
// L̄ is used (the strict-upper output entries are structural zeros). All arithmetic
// is f64 (§V10); no backend op is needed (small dense).
func init() {
	RegisterVJP(backend.OpCholesky, func(_ *backend.Context, _, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		lt := out[0] // the forward factor L (lower-triangular)
		n := lt.Shape()[0]

		// dense f64 copies: L (lower) and L̄ (lower triangle only)
		l := make([][]float64, n)
		lbar := make([][]float64, n)
		for i := range n {
			l[i] = make([]float64, n)
			lbar[i] = make([]float64, n)
			for j := 0; j <= i; j++ {
				l[i][j] = lt.AtF64(i, j)
				lbar[i][j] = g.AtF64(i, j)
			}
		}

		// P = Φ(Lᵀ·L̄): M = Lᵀ·L̄, then keep the lower triangle with a halved diagonal.
		p := make([][]float64, n)
		for i := range n {
			p[i] = make([]float64, n)
			for j := 0; j <= i; j++ {
				var m float64 // (Lᵀ·L̄)_ij = Σ_k L[k,i]·L̄[k,j]; both lower ⇒ k ≥ max(i,j)
				for k := i; k < n; k++ {
					m += l[k][i] * lbar[k][j]
				}
				if i == j {
					p[i][j] = 0.5 * m
				} else {
					p[i][j] = m
				}
			}
		}

		// Linv = L⁻¹ (lower-triangular) by forward substitution on L·X = I.
		linv := make([][]float64, n)
		for i := range n {
			linv[i] = make([]float64, n)
		}
		for j := range n {
			linv[j][j] = 1 / l[j][j]
			for i := j + 1; i < n; i++ {
				var s float64
				for k := j; k < i; k++ {
					s += l[i][k] * linv[k][j]
				}
				linv[i][j] = -s / l[i][i]
			}
		}

		// S = Linvᵀ·P·Linv. First T = P·Linv (P lower), then S = Linvᵀ·T.
		tmp := make([][]float64, n)
		for i := range n {
			tmp[i] = make([]float64, n)
			for j := range n {
				var s float64 // (P·Linv)_ij = Σ_k P[i,k]·Linv[k,j], Linv lower ⇒ k ≥ j, P lower ⇒ k ≤ i
				for k := j; k <= i; k++ {
					s += p[i][k] * linv[k][j]
				}
				tmp[i][j] = s
			}
		}
		abar := tensor.New(lt.Dtype(), lt.Shape())
		for i := range n {
			for j := range n {
				var sij float64 // S_ij = Σ_k Linvᵀ[i,k]·T[k,j] = Σ_k Linv[k,i]·T[k,j], Linv lower ⇒ k ≥ i
				var sji float64 // S_ji, Linv lower ⇒ k ≥ j
				for k := i; k < n; k++ {
					sij += linv[k][i] * tmp[k][j]
				}
				for k := j; k < n; k++ {
					sji += linv[k][j] * tmp[k][i]
				}
				// Ā = ½(S + Sᵀ): symmetric, so ⟨Ā,dA⟩=⟨L̄,dL⟩ under the tape's full
				// Frobenius inner product for any symmetric dA. Off-diagonal ½(S_ij+S_ji),
				// diagonal S_ii (½·2S_ii).
				if i == j {
					abar.SetF64(sij, i, j)
				} else {
					abar.SetF64(0.5*(sij+sji), i, j)
				}
			}
		}
		return []*tensor.Tensor{abar}, nil
	})
}
