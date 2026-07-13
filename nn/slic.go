package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SLiCLoss is the SLiC-HF sequence-likelihood-calibration loss (Zhao, Joshi, Liu, Khalman,
// Saleh, Peter J. Liu 2023, "SLiC-HF: Sequence Likelihood Calibration with Human Feedback",
// arXiv:2305.10425; the calibration objective originates in Zhao et al. 2022, "Calibrating
// Sequence Likelihood Improves Conditional Language Generation", arXiv:2210.00045). Unlike the
// LOGISTIC direct-alignment losses in this library (DPO/IPO/KTO/CPO/SimPO/ORPO, all built on
// −log σ), SLiC-HF calibrates the policy's sequence likelihoods with a MAX-MARGIN RANK (hinge)
// loss plus a cross-entropy regularizer toward the SFT target — a distinct loss geometry: the
// gradient is zero once the preferred candidate already beats the dispreferred one by the
// margin δ, instead of pushing the gap to infinity.
//
// With logPθ(y|x) the SUMMED sequence log-probability of candidate y under the policy, the
// rank calibration loss and regularization are
//
//	L_cal = mean max(0, δ − logPθ(y⁺|x) + logPθ(y⁻|x))   // hinge: preferred beats dispreferred by δ
//	L_reg = mean −logPθ(y_ref|x)                          // cross-entropy toward the SFT target
//	L     = L_cal + λ·L_reg
//
// preferred (y⁺) and dispreferred (y⁻) are [batch] tensors of summed sequence log-probs under
// the policy; δ is the positive margin (rank loss); the calibration term is differentiable
// w.r.t. both (the hinge routes gradient only when it is active, i.e. the margin is violated).
// sftTarget is the [batch] summed log-prob of the SFT/reference target y_ref; pass it with a
// positive λ to add the regularizer, or pass nil (or λ ≤ 0) for the bare calibration loss.
// Each input is the sum of the policy's per-token log-probs over that candidate (§R214, §V16).
func SLiCLoss(ctx *backend.Context, preferred, dispreferred, sftTarget *tensor.Tensor, delta, lambda float64) (*tensor.Tensor, error) {
	if !preferred.Shape().Equal(dispreferred.Shape()) {
		return nil, fmt.Errorf("nn: SLiCLoss needs equal-shaped preferred/dispreferred log-probs, got %v and %v", preferred.Shape(), dispreferred.Shape())
	}
	if sftTarget != nil && lambda > 0 && !sftTarget.Shape().Equal(preferred.Shape()) {
		return nil, fmt.Errorf("nn: SLiCLoss sftTarget shape %v must match preferred %v", sftTarget.Shape(), preferred.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// hinge argument z = δ − logPθ(y⁺) + logPθ(y⁻) = δ + (dispreferred − preferred).
	diff, err := ex(backend.OpSub, nil, dispreferred, preferred)
	if err != nil {
		return nil, err
	}
	z, err := ex(backend.OpAdd, nil, diff, scalarTensor(diff.Dtype(), delta))
	if err != nil {
		return nil, err
	}
	hinge, err := ex(backend.OpReLU, nil, z) // max(0, z)
	if err != nil {
		return nil, err
	}
	loss, err := ex(backend.OpMean, nil, hinge) // L_cal (scalar)
	if err != nil {
		return nil, err
	}
	if sftTarget == nil || lambda <= 0 {
		return loss, nil
	}
	// regularization: λ · mean(−logPθ(y_ref)); add via AXPY so the rank-0 scalar is not promoted.
	negRef, err := ex(backend.OpNeg, nil, sftTarget)
	if err != nil {
		return nil, err
	}
	reg, err := ex(backend.OpMean, nil, negRef) // L_reg (scalar)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: lambda}, reg, loss) // λ·L_reg + L_cal
}
