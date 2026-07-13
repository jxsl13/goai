package ops

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LogDet returns the natural logarithm of the determinant of a symmetric
// positive-definite matrix a, computed stably through its Cholesky factor as
// 2·Σ ln L_ii (the log-det branch of numpy.linalg.slogdet, whose sign is +1 for
// SPD input). The result is a scalar. a must be square and positive-definite; it
// errors otherwise (a non-positive Cholesky pivot).
//
// LogDet is the workhorse term of a Gaussian log-likelihood / GP marginal
// likelihood (−½·logdet(K) − …) and of normalizing-flow log-densities. It is
// differentiable: by Jacobi's formula ∂ logdet(A) / ∂A = A⁻ᵀ, which is A⁻¹ for
// symmetric A (computed from the same Cholesky factor), so gradients flow to
// whatever produced the SPD matrix.
func LogDet(a *tensor.Tensor) (*tensor.Tensor, error) {
	return eager(backend.OpLogDet, a)
}
