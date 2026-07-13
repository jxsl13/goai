package ops

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Cholesky returns the lower-triangular Cholesky factor L of a symmetric
// positive-definite matrix a, with a = L·Lᵀ (numpy.linalg.cholesky). a must be a
// square matrix; only its lower triangle is read (the upper is assumed symmetric,
// matching LAPACK potrf/numpy's uplo='L'). L is lower-triangular with a positive
// diagonal and exact zeros above it, so it factors a into a "matrix square root"
// used for sampling correlated Gaussians, whitening, and solving SPD systems.
//
// It errors if a is non-square or not positive-definite (a non-positive pivot).
//
// It is differentiable: the gradient w.r.t. a is the reverse-mode Cholesky of
// Murray 2016 (arXiv:1602.07527) — the symmetric Ā = ½(S + Sᵀ) with
// S = L⁻ᵀ·Φ(Lᵀ·L̄)·L⁻¹ and Φ the lower triangle with a halved diagonal (the
// PyTorch/JAX symmetrization, so it composes correctly on the autograd tape).
func Cholesky(a *tensor.Tensor) (*tensor.Tensor, error) {
	return eager(backend.OpCholesky, a)
}
