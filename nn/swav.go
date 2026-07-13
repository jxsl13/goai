package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SwAVTemperature is the default softmax temperature τ of the SwAV prediction (§3, τ=0.1).
const SwAVTemperature = 0.1

// SwAVLoss computes the SwAV swapped-prediction loss (Caron, Misra, Mairal, Goyal,
// Bojanowski & Joulin 2020, "Unsupervised Learning of Visual Features by Contrasting
// Cluster Assignments", NeurIPS, arXiv:2006.09882, §3 Eq. 1/2). SwAV is a clustering-based
// self-supervised objective — distinct from the instance-contrastive InfoNCE, the
// redundancy-reduction Barlow Twins, and the variance-covariance VICReg already here: two
// augmented views of an image are projected onto K trainable PROTOTYPES, an online
// cluster assignment ("code") is computed for each view with Sinkhorn-Knopp (see Sinkhorn;
// equipartitioned, detached), and each view's features must predict the OTHER view's code:
//
//	L = ℓ(z_t, q_s) + ℓ(z_s, q_t),   ℓ(z, q) = − Σ_k q^(k) · log softmax(z·Cᵀ/τ)^(k)
//
// This "swapped prediction" (view t predicts s's code and vice versa) is what makes the
// prototypes and features consistent across augmentations without negatives.
//
// scoresT and scoresS are the two views' prototype scores z·Cᵀ ∈ ℝ^{B×K} (batch × K
// prototypes); codeT and codeS are their Sinkhorn codes (soft assignments, [B,K]), which
// are fixed TARGETS — no gradient flows through them. temp is the softmax temperature τ.
// The loss is differentiable w.r.t. the two score matrices (hence the features and
// prototypes). All four inputs share shape [B,K].
func SwAVLoss(ctx *backend.Context, scoresT, scoresS, codeT, codeS *tensor.Tensor, temp float64) (*tensor.Tensor, error) {
	for _, m := range []*tensor.Tensor{scoresT, scoresS, codeT, codeS} {
		if m.Ndim() != 2 {
			return nil, fmt.Errorf("nn: SwAVLoss wants rank-2 [B,K] inputs, got %v", m.Shape())
		}
		if !m.Shape().Equal(scoresT.Shape()) {
			return nil, fmt.Errorf("nn: SwAVLoss shape mismatch: %v vs %v", m.Shape(), scoresT.Shape())
		}
	}
	if temp <= 0 {
		return nil, fmt.Errorf("nn: SwAVLoss temperature must be > 0, got %g", temp)
	}
	// swapped prediction: view t predicts s's code and vice versa.
	lt, err := swavSoftCE(ctx, scoresT, codeS, temp)
	if err != nil {
		return nil, err
	}
	ls, err := swavSoftCE(ctx, scoresS, codeT, temp)
	if err != nil {
		return nil, err
	}
	o, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{lt, ls}, nil)
	if err != nil {
		return nil, err
	}
	return o[0], nil
}

// swavSoftCE is the soft-target cross-entropy mean_b(−Σ_k target[b,k]·log softmax(logits[b]/τ)_k),
// with a numerically stable log-softmax (subtract the row max before exp). Differentiable
// w.r.t. logits; target is a fixed distribution.
func swavSoftCE(ctx *backend.Context, logits, target *tensor.Tensor, temp float64) (*tensor.Tensor, error) {
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	b := logits.Shape()[0]
	lastAxis := backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}
	scaled, err := ex(backend.OpMul, nil, logits, scalarTensor(logits.Dtype(), 1/temp)) // logits/τ
	if err != nil {
		return nil, err
	}
	// stable log-softmax: logp = (scaled − max) − log Σ exp(scaled − max).
	mx, err := ex(backend.OpMax, lastAxis, scaled)
	if err != nil {
		return nil, err
	}
	shifted, err := ex(backend.OpSub, nil, scaled, mx) // broadcast [B,K]-[B,1]
	if err != nil {
		return nil, err
	}
	expv, err := ex(backend.OpExp, nil, shifted)
	if err != nil {
		return nil, err
	}
	sumexp, err := ex(backend.OpSum, lastAxis, expv)
	if err != nil {
		return nil, err
	}
	lse, err := ex(backend.OpLog, nil, sumexp)
	if err != nil {
		return nil, err
	}
	logp, err := ex(backend.OpSub, nil, shifted, lse) // [B,K] − [B,1]
	if err != nil {
		return nil, err
	}
	prod, err := ex(backend.OpMul, nil, target, logp) // q ⊙ log p
	if err != nil {
		return nil, err
	}
	// −(1/B)·Σ_{b,k} q·log p, folding the −1/B into a pre-reduction scale.
	scaledProd, err := ex(backend.OpMul, nil, prod, scalarTensor(logits.Dtype(), -1/float64(b)))
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, scaledProd)
}
