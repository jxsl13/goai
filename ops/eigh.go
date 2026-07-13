package ops

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Eigh computes the eigendecomposition of a real symmetric matrix a,
// a = V·diag(w)·Vᵀ (numpy.linalg.eigh): w holds the eigenvalues in ascending order
// and the columns of V are the corresponding orthonormal eigenvectors (VᵀV = I,
// a·V[:,k] = w[k]·V[:,k]). a must be square; its symmetric part (a+aᵀ)/2 is used.
//
// Eigh underlies PCA, spectral penalties, whitening, and any layer that needs the
// spectrum of a symmetric matrix. It is differentiable in a (Ionescu et al. 2015):
// with cotangents w̄, V̄, the gradient is the symmetric
// Ā = ½(G + Gᵀ) with G = V·(diag(w̄) + F ∘ (Vᵀ·V̄))·Vᵀ and F_ij = 1/(w_j − w_i)
// (0 on the diagonal), so gradients flow to whatever produced a. The eigenvector
// gradient assumes distinct eigenvalues (F is singular at a repeated eigenvalue).
func Eigh(a *tensor.Tensor) (w, v *tensor.Tensor, err error) {
	out, err := backend.Execute(backend.NewContext(), backend.OpEigh, []*tensor.Tensor{a}, nil)
	if err != nil {
		return nil, nil, err
	}
	return out[0], out[1], nil
}
