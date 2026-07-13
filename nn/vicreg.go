package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

const (
	vicGamma = 1.0  // variance target γ
	vicEps   = 1e-4 // ε inside the regularized std √(Var+ε)
)

// VICRegLoss computes the VICReg self-supervised loss (Bardes, Ponce & LeCun 2021/2022
// ICLR, "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised
// Learning", arXiv:2105.04906). Like Barlow Twins it needs no negatives, but it
// prevents representation collapse with three EXPLICIT regularizers applied to two
// views za, zb ∈ ℝ^{N×D}:
//
//   - Invariance  s = (1/N)·Σ_i ‖z_i − z'_i‖²           (pull the two views together)
//   - Variance    v(Z) = (1/D)·Σ_j max(0, γ − √(Var_j+ε))  (keep every dim's std ≥ γ:
//     a hinge that directly blocks collapse)
//   - Covariance  c(Z) = (1/D)·Σ_{i≠j} Cov(Z)_ij²        (decorrelate the dimensions)
//
// The total is ℓ = λ·s + μ·[v(za)+v(zb)] + ν·[c(za)+c(zb)] (the paper's defaults are
// λ=μ=25, ν=1). γ=1, ε=1e-4, and the variance/covariance use the unbiased (N−1)
// estimator, matching the reference implementation. za, zb are [N, D]; the loss is
// differentiable w.r.t. both.
func VICRegLoss(ctx *backend.Context, za, zb *tensor.Tensor, lambda, mu, nu float64) (*tensor.Tensor, error) {
	if za.Ndim() != 2 || zb.Ndim() != 2 {
		return nil, fmt.Errorf("nn: VICRegLoss wants rank-2 [N,D], got %v, %v", za.Shape(), zb.Shape())
	}
	if !za.Shape().Equal(zb.Shape()) {
		return nil, fmt.Errorf("nn: VICRegLoss shape mismatch: %v vs %v", za.Shape(), zb.Shape())
	}
	n := za.Shape()[0]
	if n < 2 {
		return nil, fmt.Errorf("nn: VICRegLoss needs batch N≥2 for the (N−1) variance, got %d", n)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	allAxes := []int{0, 1}
	// invariance s = (1/N)Σ‖za−zb‖²; fold λ/N into the elementwise scale, then sum.
	diff, err := ex(backend.OpSub, nil, za, zb)
	if err != nil {
		return nil, err
	}
	sq, err := ex(backend.OpMul, nil, diff, diff)
	if err != nil {
		return nil, err
	}
	sScaled, err := ex(backend.OpMul, nil, sq, scalarTensor(za.Dtype(), lambda/float64(n)))
	if err != nil {
		return nil, err
	}
	total, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: allAxes}, sScaled)
	if err != nil {
		return nil, err
	}
	for _, z := range []*tensor.Tensor{za, zb} {
		v, err := vicVarianceTerm(ctx, z, mu)
		if err != nil {
			return nil, err
		}
		c, err := vicCovarianceTerm(ctx, z, nu)
		if err != nil {
			return nil, err
		}
		if total, err = ex(backend.OpAdd, nil, total, v); err != nil {
			return nil, err
		}
		if total, err = ex(backend.OpAdd, nil, total, c); err != nil {
			return nil, err
		}
	}
	return total, nil
}

// vicVarianceTerm returns coeff·v(Z) = coeff·(1/D)Σ_j max(0, γ − √(Var_j+ε)) as a
// scalar, with the (N−1)-unbiased per-dimension batch variance.
func vicVarianceTerm(ctx *backend.Context, z *tensor.Tensor, coeff float64) (*tensor.Tensor, error) {
	n, d := z.Shape()[0], z.Shape()[1]
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	centered, err := vicCenter(ctx, z)
	if err != nil {
		return nil, err
	}
	sq, err := ex(backend.OpMul, nil, centered, centered)
	if err != nil {
		return nil, err
	}
	ss, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0}, KeepDims: true}, sq) // [1,D]
	if err != nil {
		return nil, err
	}
	varEps, err := ex(backend.OpMul, nil, ss, scalarTensor(z.Dtype(), 1/float64(n-1))) // Var (unbiased)
	if err != nil {
		return nil, err
	}
	epsT := tensor.New(z.Dtype(), tensor.Shape{1, 1})
	epsT.SetF64(vicEps, 0, 0)
	if varEps, err = ex(backend.OpAdd, nil, varEps, epsT); err != nil { // Var+ε
		return nil, err
	}
	std, err := ex(backend.OpSqrt, nil, varEps)
	if err != nil {
		return nil, err
	}
	gammaT := tensor.New(z.Dtype(), tensor.Shape{1, 1})
	gammaT.SetF64(vicGamma, 0, 0)
	gap, err := ex(backend.OpSub, nil, gammaT, std) // γ − std (broadcast [1,1]-[1,D])
	if err != nil {
		return nil, err
	}
	hinge, err := ex(backend.OpReLU, nil, gap) // max(0, γ−std)
	if err != nil {
		return nil, err
	}
	scaled, err := ex(backend.OpMul, nil, hinge, scalarTensor(z.Dtype(), coeff/float64(d))) // coeff·(1/D)Σ
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, scaled)
}

// vicCovarianceTerm returns coeff·c(Z) = coeff·(1/D)Σ_{i≠j} Cov_ij² as a scalar.
func vicCovarianceTerm(ctx *backend.Context, z *tensor.Tensor, coeff float64) (*tensor.Tensor, error) {
	n, d := z.Shape()[0], z.Shape()[1]
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	centered, err := vicCenter(ctx, z)
	if err != nil {
		return nil, err
	}
	ct, err := ex(backend.OpTranspose, nil, centered) // [D,N]
	if err != nil {
		return nil, err
	}
	cov, err := ex(backend.OpMatMul, nil, ct, centered) // [D,D] = Σ(z−z̄)(z−z̄)ᵀ
	if err != nil {
		return nil, err
	}
	if cov, err = ex(backend.OpMul, nil, cov, scalarTensor(z.Dtype(), 1/float64(n-1))); err != nil { // /(N−1)
		return nil, err
	}
	covSq, err := ex(backend.OpMul, nil, cov, cov)
	if err != nil {
		return nil, err
	}
	// zero the diagonal, keep the off-diagonal squared entries.
	offMask := tensor.New(z.Dtype(), tensor.Shape{d, d})
	for i := range d {
		for j := range d {
			if i != j {
				offMask.SetF64(1, i, j)
			}
		}
	}
	off, err := ex(backend.OpMul, nil, covSq, offMask)
	if err != nil {
		return nil, err
	}
	scaled, err := ex(backend.OpMul, nil, off, scalarTensor(z.Dtype(), coeff/float64(d)))
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, scaled)
}

// vicCenter subtracts the per-dimension batch mean from z [N,D].
func vicCenter(ctx *backend.Context, z *tensor.Tensor) (*tensor.Tensor, error) {
	mean, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{z}, backend.ReduceAttrs{Axes: []int{0}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	o, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{z, mean[0]}, nil)
	if err != nil {
		return nil, err
	}
	return o[0], nil
}
