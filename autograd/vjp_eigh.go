package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Eigh VJP (Ionescu, Vantzos & Sminchisescu 2015 "Matrix Backpropagation",
// arXiv:1509.07838; Giles 2008; PyTorch linalg_eigh_backward). For the symmetric
// A = V·diag(w)·Vᵀ (w ascending, V orthonormal columns) with output cotangents w̄
// (eigenvalues) and V̄ (eigenvectors):
//
//	F_ij = 1/(w_j − w_i) for i≠j, 0 on the diagonal   (Giles/PyTorch sign)
//	G    = V·(diag(w̄) + F ∘ (Vᵀ·V̄))·Vᵀ
//	Ā    = ½(G + Gᵀ)
//
// The ½(G+Gᵀ) symmetrization gives the gradient for the symmetric input under the
// tape's full Frobenius product. The eigenvalue term V·diag(w̄)·Vᵀ is exact and
// degeneracy-free; the eigenvector term needs DISTINCT eigenvalues (F blows up at a
// repeated eigenvalue). MULTI-output VJP: gouts = [w̄, V̄] (zero where unused). All
// arithmetic is f64 (§V10).
func init() {
	RegisterVJPMulti(backend.OpEigh, func(_ *backend.Context, _, outputs []*tensor.Tensor, _ backend.Attrs, gouts []*tensor.Tensor) ([]*tensor.Tensor, error) {
		wt, vt := outputs[0], outputs[1]
		wbar, vbar := gouts[0], gouts[1]
		n := wt.Shape()[0]

		w := make([]float64, n)
		wb := make([]float64, n)
		for i := range n {
			w[i] = wt.AtF64(i)
			wb[i] = wbar.AtF64(i)
		}
		v := to2D(vt, n, n)
		vb := to2D(vbar, n, n)
		// Both operands of the inner product below are read DOWN a column, and at these sizes a
		// row is its own page-sized allocation, so a column walk misses on nearly every element.
		// Transposed copies make each inner loop walk two contiguous runs; the transpose itself is
		// O(n^2) against the O(n^3) it feeds. Bit-identical: only where an operand lives changes,
		// never which operand or in what order (the same argument the Cholesky VJP records).
		vT := make([][]float64, n)
		vbT := make([][]float64, n)
		for i := range n {
			vT[i], vbT[i] = make([]float64, n), make([]float64, n)
		}
		for r := range n {
			vr, vbr := v[r], vb[r]
			for c := range n {
				vT[c][r], vbT[c][r] = vr[c], vbr[c]
			}
		}

		// inner = diag(w̄) + F ∘ (Vᵀ·V̄).
		// The three n^3 products below are the whole cost of this rule — 1.28s of a 1.48s profile
		// at n=256 — and each has an independent outer index writing only its own row, so a
		// striped split leaves every element accumulating its own terms in its own order. Rows are
		// allocated before the split so the workers only write into them.
		inner := make([][]float64, n)
		for i := range n {
			inner[i] = make([]float64, n)
		}
		logdetParallelIdx(n, n*n*n, func(i int) {
			vTi := vT[i]
			for j := range n {
				var p float64 // (Vᵀ·V̄)_ij = Σ_r V[r,i]·V̄[r,j]
				vbTj := vbT[j]
				for r := range n {
					p += vTi[r] * vbTj[r]
				}
				if i != j {
					inner[i][j] = p / (w[j] - w[i]) // F_ij ∘ P_ij, F_ij = 1/(w_j − w_i)
				}
			}
			inner[i][i] += wb[i]
		})
		// G = V·inner·Vᵀ  (T = inner·Vᵀ, then G = V·T).
		tmp := make([][]float64, n)
		for a := range n {
			tmp[a] = make([]float64, n)
		}
		logdetParallelIdx(n, n*n*n, func(a int) {
			for j := range n {
				var s float64 // (inner·Vᵀ)_aj = Σ_b inner[a,b]·V[j,b]
				for b := range n {
					s += inner[a][b] * v[j][b]
				}
				tmp[a][j] = s
			}
		})
		abar := tensor.New(vt.Dtype(), tensor.Shape{n, n})
		// Only the UPPER triangle is computed, and each off-diagonal result is mirrored. The full
		// double loop formed every pair TWICE: at (j,i) its g is this (i,j)'s gt and vice versa, so
		// half of the O(n^3) accumulation was recomputing sums it had already made, and on the
		// diagonal it formed the identical sum twice and discarded one.
		//
		// BIT-IDENTICAL, on the argument the Cholesky and SolveSPD VJPs already record: the full
		// loop stored 0.5*(gt+g) at (j,i) where this stores 0.5*(g+gt), and IEEE addition is
		// commutative — a+b and b+a have the same bits for all non-NaN operands, which these are by
		// construction. Each individual sum keeps its own ascending-a order and operands.
		// tmp IS READ DOWN ITS COLUMNS BELOW, one element from each of n separately allocated
		// rows per inner step, and the triangular loop does that n²/2 times. Mirroring it
		// transposed costs one O(n²) pass and makes both inner reads contiguous. Pure copy —
		// every value is the same float64 from a different address.
		tmpT := make([][]float64, n)
		for j := range n {
			row := make([]float64, n)
			for a := range n {
				row[a] = tmp[a][j]
			}
			tmpT[j] = row
		}
		// The triangular loop bands too. Index i writes (i,j) and (j,i) for j ≥ i, so two
		// different i values never write the same cell: a clash would force them equal. Each
		// entry keeps its own ascending-a accumulation, so the split is bit-identical.
		logdetParallelIdx(n, n*n*n/2, func(i int) {
			vi := v[i]
			{
				for j := i; j < n; j++ {
					var g, gt float64 // G[i,j] and G[j,i]
					vj := v[j]
					tj, ti := tmpT[j], tmpT[i]
					for a, va := range vi {
						g += va * tj[a]
						gt += vj[a] * ti[a]
					}
					if i == j {
						abar.SetF64(0.5*(g+gt), i, j)
						continue
					}
					m := 0.5 * (g + gt) // Ā = ½(G+Gᵀ) is symmetric
					abar.SetF64(m, i, j)
					abar.SetF64(m, j, i)
				}
			}
		})
		return []*tensor.Tensor{abar}, nil
	})
}
