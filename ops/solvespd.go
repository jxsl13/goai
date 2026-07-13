package ops

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SolveSPD solves the symmetric positive-definite linear system a·X = b for X,
// i.e. X = a⁻¹·b, via the Cholesky factor of a (numpy.linalg.solve restricted to
// SPD a). b is a right-hand-side vector [n] or matrix [n,k]; X has b's shape. a
// must be square and positive-definite, else it errors (a non-positive Cholesky
// pivot). Solving through the factor is ~2× cheaper than forming a⁻¹ and is the
// standard way to compute α = K⁻¹y in a Gaussian process.
//
// It is differentiable in both a and b (standard reverse-mode of a linear solve):
// the cotangent of b is B̄ = a⁻¹·Ḡ (another SPD solve), and that of a is the
// symmetric Ā = −½(B̄·Xᵀ + X·B̄ᵀ), so gradients flow to whatever produced the
// matrix or the right-hand side.
func SolveSPD(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	return eager(backend.OpSolveSPD, a, b)
}
