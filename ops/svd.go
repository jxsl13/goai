package ops

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SVD computes the reduced (economy) singular value decomposition a = U·diag(s)·Vᵀ
// of a matrix a with m ≥ n rows (numpy.linalg.svd full_matrices=False): U ∈ Rᵐˣⁿ and
// V ∈ Rⁿˣⁿ have orthonormal columns and s holds the singular values σ₁ ≥ … ≥ σₙ ≥ 0.
// a must be a tall or square rank-2 matrix, else it errors.
//
// SVD is the tool behind low-rank approximation, the pseudoinverse, PCA, and the
// nuclear/spectral norms. It is differentiable in a (Townsend 2016; Ionescu et al.
// 2015): with cotangents Ū, s̄, V̄, gradients flow to whatever produced a. The
// singular-value gradient (Ā = U·diag(s̄)·Vᵀ) is exact; the U/V gradient assumes
// distinct, nonzero singular values.
func SVD(a *tensor.Tensor) (u, s, v *tensor.Tensor, err error) {
	out, err := backend.Execute(backend.NewContext(), backend.OpSVD, []*tensor.Tensor{a}, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	return out[0], out[1], out[2], nil
}
