package ops

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// QR computes the reduced (economy) QR factorization of a matrix a with m ≥ n rows:
// a = Q·R with Q ∈ Rᵐˣⁿ having orthonormal columns (QᵀQ = Iₙ) and R ∈ Rⁿˣⁿ
// upper-triangular (numpy.linalg.qr mode='reduced'), via backward-stable Householder
// reflectors. a must be a tall or square rank-2 matrix, else it errors.
//
// QR is the standard tool for orthonormalizing a set of columns and for stable
// least-squares. It is differentiable in a (Seeger et al. 2017): with output
// cotangents Q̄, R̄, the gradient is Ā = [Q̄ + Q·copyltu(R·R̄ᵀ − Q̄ᵀ·Q)]·R⁻ᵀ, so
// gradients flow to whatever produced a.
func QR(a *tensor.Tensor) (q, r *tensor.Tensor, err error) {
	out, err := backend.Execute(backend.NewContext(), backend.OpQR, []*tensor.Tensor{a}, nil)
	if err != nil {
		return nil, nil, err
	}
	return out[0], out[1], nil
}
