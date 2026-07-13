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

		// inner = diag(w̄) + F ∘ (Vᵀ·V̄).
		inner := make([][]float64, n)
		for i := range n {
			inner[i] = make([]float64, n)
			for j := range n {
				var p float64 // (Vᵀ·V̄)_ij = Σ_r V[r,i]·V̄[r,j]
				for r := range n {
					p += v[r][i] * vb[r][j]
				}
				if i != j {
					inner[i][j] = p / (w[j] - w[i]) // F_ij ∘ P_ij, F_ij = 1/(w_j − w_i)
				}
			}
			inner[i][i] += wb[i]
		}
		// G = V·inner·Vᵀ  (T = inner·Vᵀ, then G = V·T).
		tmp := make([][]float64, n)
		for a := range n {
			tmp[a] = make([]float64, n)
			for j := range n {
				var s float64 // (inner·Vᵀ)_aj = Σ_b inner[a,b]·V[j,b]
				for b := range n {
					s += inner[a][b] * v[j][b]
				}
				tmp[a][j] = s
			}
		}
		abar := tensor.New(vt.Dtype(), tensor.Shape{n, n})
		for i := range n {
			for j := range n {
				var g, gt float64 // G[i,j] and G[j,i]
				for a := range n {
					g += v[i][a] * tmp[a][j]
					gt += v[j][a] * tmp[a][i]
				}
				abar.SetF64(0.5*(g+gt), i, j) // Ā = ½(G+Gᵀ)
			}
		}
		return []*tensor.Tensor{abar}, nil
	})
}
