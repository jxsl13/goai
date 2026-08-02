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

		// EVERY ONE OF THE FOUR CUBIC LOOPS BELOW READS ITS OPERANDS ALONG A COLUMN in the obvious
		// row-major layout, and at these sizes that is the dominant cost: a row is make([]float64,
		// n), so at n=512 consecutive rows sit exactly one 4096-byte page apart and a column walk
		// maps into a handful of L1 sets, missing on nearly every element. The intermediates are
		// therefore built TRANSPOSED — lT, lbarT, linvT, tmpT hold columns as contiguous rows — so
		// each inner loop walks two contiguous runs.
		//
		// BIT-IDENTICAL: only the storage layout of intermediates changes. Every sum keeps its own
		// operands and its ascending-k order, and no accumulator is split.
		l := make([][]float64, n)     // row-major L: the substitution below reads L by ROW
		lT := make([][]float64, n)    // lT[i][k] = L[k,i]
		lbarT := make([][]float64, n) // lbarT[j][k] = L̄[k,j]
		for i := range n {
			l[i], lT[i], lbarT[i] = make([]float64, n), make([]float64, n), make([]float64, n)
		}
		for i := range n {
			li := l[i]
			for j := 0; j <= i; j++ {
				v := lt.AtF64(i, j)
				li[j] = v
				lT[j][i] = v
				lbarT[j][i] = g.AtF64(i, j)
			}
		}

		// P = Φ(Lᵀ·L̄): M = Lᵀ·L̄, then keep the lower triangle with a halved diagonal.
		p := make([][]float64, n)
		for i := range n {
			p[i] = make([]float64, n)
		}
		// Row i of P depends only on lT[i] and lbarT, and writes only p[i] — disjoint across i, so
		// the striped schedule is bit-identical to the serial loop (each sum keeps its own
		// ascending-k order). Same argument the logdet VJP records for its Ā loop.
		logdetParallelIdx(n, n*n*n, func(i int) {
			pi := p[i]
			lTi := lT[i]
			for j := 0; j <= i; j++ {
				var m float64 // (Lᵀ·L̄)_ij = Σ_k L[k,i]·L̄[k,j]; both lower ⇒ k ≥ max(i,j)
				lbTj := lbarT[j]
				for k := i; k < n; k++ {
					m += lTi[k] * lbTj[k]
				}
				if i == j {
					pi[j] = 0.5 * m
				} else {
					pi[j] = m
				}
			}
		})

		// Linv = L⁻¹ (lower-triangular) by forward substitution on L·X = I, solved straight into
		// column-major form: linvT[j] IS column j of Linv, which is exactly what the inner sum
		// reads and what both consumers below want.
		linvT := make([][]float64, n)
		for i := range n {
			linvT[i] = make([]float64, n)
		}
		// Columns are independent: column j writes only linvT[j] and reads the factor, so a split
		// leaves every entry accumulating its own terms in its own order. The work is triangular —
		// column j costs about (n-j)^2/2 — so the STRIPED helper is the right one; equal bands
		// would give the first worker the whole wide end. Total work is about n^3/6, which is what
		// the gate must see (a smaller estimate leaves mid-sized cases serial, §T1083).
		logdetParallelIdx(n, n*n*n/6, func(j int) {
			cj := linvT[j]
			cj[j] = 1 / l[j][j]
			for i := j + 1; i < n; i++ {
				li := l[i] // invariant in k — one pointer load instead of i-j (PS4006)
				var s float64
				for k := j; k < i; k++ {
					s += li[k] * cj[k]
				}
				cj[i] = -s / li[i]
			}
		})

		// S = Linvᵀ·P·Linv. First T = P·Linv (P lower), then S = Linvᵀ·T. T is built transposed
		// with j outermost so each tmpT[j] fills contiguously and both operands of the inner sum
		// are contiguous runs.
		tmpT := make([][]float64, n)
		for j := range n {
			tmpT[j] = make([]float64, n)
		}
		// Column j of T reads linvT[j] and P and writes only tmpT[j] — disjoint across j.
		logdetParallelIdx(n, n*n*n, func(j int) {
			tj := tmpT[j]
			cj := linvT[j]
			for i := range n {
				var s float64 // (P·Linv)_ij = Σ_k P[i,k]·Linv[k,j], Linv lower ⇒ k ≥ j, P lower ⇒ k ≤ i
				pi := p[i]
				for k := j; k <= i; k++ {
					s += pi[k] * cj[k]
				}
				tj[i] = s
			}
		})
		abar := tensor.New(lt.Dtype(), lt.Shape())
		// Only the UPPER triangle is computed, and each off-diagonal result is mirrored. The full
		// double loop did every pair twice: at (j,i) its sij is this (i,j)'s sji and vice versa, so
		// half of the O(n^3) accumulation was recomputing sums it had already formed — and at i==j
		// it formed the identical sum twice and discarded one.
		//
		// BIT-IDENTICAL, on the same argument vjp_solvespd already records for this transform: the
		// full loop stored 0.5*(sji+sij) at (j,i) where this stores 0.5*(sij+sji), and IEEE
		// addition is commutative — a+b and b+a have the same bits for all non-NaN operands, which
		// these are by construction. Each individual sum keeps its own ascending-k order and
		// operands, so nothing is reassociated.
		// Iteration i writes only (i,j) and (j,i) for j >= i. Two different i cannot collide: a
		// clash would need i1 == j' and j == i2 with j >= i1 and j' >= i2, which forces i1 == i2.
		logdetParallelIdx(n, n*n*n, func(i int) {
			ci, ti := linvT[i], tmpT[i]
			for j := i; j < n; j++ {
				var sij float64 // S_ij = Σ_k Linvᵀ[i,k]·T[k,j] = Σ_k Linv[k,i]·T[k,j], Linv lower ⇒ k ≥ i
				var sji float64 // S_ji, Linv lower ⇒ k ≥ j
				cj, tj := linvT[j], tmpT[j]
				for k := i; k < n; k++ {
					sij += ci[k] * tj[k]
				}
				for k := j; k < n; k++ {
					sji += cj[k] * ti[k]
				}
				// Ā = ½(S + Sᵀ): symmetric, so ⟨Ā,dA⟩=⟨L̄,dL⟩ under the tape's full
				// Frobenius inner product for any symmetric dA. Off-diagonal ½(S_ij+S_ji),
				// diagonal S_ii (½·2S_ii).
				if i == j {
					abar.SetF64(sij, i, j)
				} else {
					v := 0.5 * (sij + sji)
					abar.SetF64(v, i, j)
					abar.SetF64(v, j, i)
				}
			}
		})
		return []*tensor.Tensor{abar}, nil
	})
}
