package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// barlowEps matches the reference implementation's BatchNorm1d(affine=False)
// standardization epsilon.
const barlowEps = 1e-5

// BarlowTwinsLoss computes the Barlow Twins self-supervised loss (Zbontar, Jing,
// Misra, LeCun & Deny 2021, "Barlow Twins: Self-Supervised Learning via Redundancy
// Reduction", ICML, arXiv:2103.03230). Given two distorted views of a batch encoded
// into embeddings za, zb ∈ ℝ^{N×D}, it makes the two views agree WITHOUT negative
// pairs — avoiding the representation collapse that plagues negative-free methods —
// by pushing their cross-correlation matrix toward the identity:
//
//	L = Σ_i (1 − C_ii)² + λ · Σ_i Σ_{j≠i} C_ij²
//
// where C = (Ẑ^A)ᵀ Ẑ^B / N is the D×D cross-correlation between the two views along
// the batch, computed after standardizing each embedding dimension to mean 0 / unit
// std over the batch. The invariance term (1−C_ii)² makes each feature perfectly
// correlated across the two views; the redundancy-reduction term λ·C_ij² decorrelates
// distinct features so they carry non-redundant information. λ defaults to 0.005.
//
// za, zb are [N, D]; the returned scalar loss is differentiable w.r.t. both.
func BarlowTwinsLoss(ctx *backend.Context, za, zb *tensor.Tensor, lambda float64) (*tensor.Tensor, error) {
	if za.Ndim() != 2 || zb.Ndim() != 2 {
		return nil, fmt.Errorf("nn: BarlowTwinsLoss wants rank-2 [N,D], got %v, %v", za.Shape(), zb.Shape())
	}
	if !za.Shape().Equal(zb.Shape()) {
		return nil, fmt.Errorf("nn: BarlowTwinsLoss shape mismatch: %v vs %v", za.Shape(), zb.Shape())
	}
	n, d := za.Shape()[0], za.Shape()[1]
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	na, err := barlowStandardize(ctx, za)
	if err != nil {
		return nil, err
	}
	nb, err := barlowStandardize(ctx, zb)
	if err != nil {
		return nil, err
	}
	naT, err := ex(backend.OpTranspose, nil, na) // [D,N]
	if err != nil {
		return nil, err
	}
	cross, err := ex(backend.OpMatMul, nil, naT, nb) // [D,D]
	if err != nil {
		return nil, err
	}
	c, err := ex(backend.OpMul, nil, cross, scalarTensor(za.Dtype(), 1/float64(n))) // C = ẐᴬᵀẐᴮ/N
	if err != nil {
		return nil, err
	}
	// L = Σ_ij W_ij·(C_ij − I_ij)² with W = 1 on the diagonal, λ off it.
	eye := tensor.New(za.Dtype(), tensor.Shape{d, d})
	weight := tensor.New(za.Dtype(), tensor.Shape{d, d})
	for i := range d {
		//perfscan:ignore PS1001 constant eye/weight fill, matmul-dominated loss
		for j := range d {
			if i == j {
				eye.SetF64(1, i, j)
				weight.SetF64(1, i, j)
			} else {
				weight.SetF64(lambda, i, j)
			}
		}
	}
	diff, err := ex(backend.OpSub, nil, c, eye)
	if err != nil {
		return nil, err
	}
	sq, err := ex(backend.OpMul, nil, diff, diff)
	if err != nil {
		return nil, err
	}
	weighted, err := ex(backend.OpMul, nil, sq, weight)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, weighted)
}

// barlowStandardize normalizes each feature (column) of x [N,D] to mean 0 / unit std
// over the batch (per-feature BatchNorm without affine), differentiably.
func barlowStandardize(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	mean, err := ex(backend.OpMean, backend.ReduceAttrs{Axes: []int{0}, KeepDims: true}, x) // [1,D]
	if err != nil {
		return nil, err
	}
	centered, err := ex(backend.OpSub, nil, x, mean)
	if err != nil {
		return nil, err
	}
	sq, err := ex(backend.OpMul, nil, centered, centered)
	if err != nil {
		return nil, err
	}
	variance, err := ex(backend.OpMean, backend.ReduceAttrs{Axes: []int{0}, KeepDims: true}, sq) // biased
	if err != nil {
		return nil, err
	}
	epsT := tensor.New(x.Dtype(), tensor.Shape{1, 1})
	epsT.SetF64(barlowEps, 0, 0)
	varEps, err := ex(backend.OpAdd, nil, variance, epsT)
	if err != nil {
		return nil, err
	}
	std, err := ex(backend.OpSqrt, nil, varEps)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpDiv, nil, centered, std)
}
