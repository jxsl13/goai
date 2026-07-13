package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// FocalLoss is the multi-class softmax focal loss (Lin, Goyal, Girshick, He & Dollár 2018, "Focal
// Loss for Dense Object Detection" / RetinaNet, arXiv:1708.02002, Eq. 4/5). It addresses class
// imbalance by DOWN-WEIGHTING well-classified examples so training concentrates on the hard ones.
// With p_t = softmax(logits)[ground-truth class] the per-example loss is
//
//	FL(p_t) = −α·(1−p_t)^γ·log(p_t)
//
// The modulating factor (1−p_t)^γ shrinks the loss of confident-correct examples (p_t→1 ⇒ factor
// →0) while leaving misclassified ones (p_t small) nearly untouched; γ≥0 is the focusing strength
// (γ=0 recovers ordinary cross-entropy −α·log p_t) and α a class-balancing weight. Paper defaults
// γ=2, α=0.25.
//
// The paper defines focal loss for BINARY detection with a sigmoid p_t (Eq. 4, and the α-balanced
// Eq. 5); this is the standard MULTI-CLASS SOFTMAX generalization (p_t = softmax prob of the true
// class), the natural extension used across the community (the paper/torchvision themselves use the
// binary sigmoid form).
//
// logits is [batch, classes] and targets is [batch] of class indices (like nn.CrossEntropy). The
// loss is the batch mean and is differentiable w.r.t. the logits (targets are constant). Built by
// ex-composition — (1−p_t)^γ = exp(γ·log(1−p_t)) since there is no power op.
func FocalLoss(ctx *backend.Context, logits, targets *tensor.Tensor, gamma, alpha float64) (*tensor.Tensor, error) {
	if logits.Ndim() != 2 {
		return nil, fmt.Errorf("nn: FocalLoss wants rank-2 [batch,classes] logits, got %v", logits.Shape())
	}
	batch, classes := logits.Shape()[0], logits.Shape()[1]
	if targets.Numel() != batch {
		return nil, fmt.Errorf("nn: FocalLoss needs %d targets for batch %d, got %d", batch, batch, targets.Numel())
	}
	// one-hot of the (constant) targets, [batch, classes].
	oneHot := tensor.New(logits.Dtype(), tensor.Shape{batch, classes})
	for i := range batch {
		c := int(targets.AtF64(tensor.Unravel(i, targets.Shape())...))
		if c < 0 || c >= classes {
			return nil, fmt.Errorf("nn: FocalLoss target %d out of range [0,%d)", c, classes)
		}
		oneHot.SetF64(1, i, c)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	lastAxis := backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}

	p, err := ex(backend.OpSoftmax, nil, logits) // softmax over the last axis (classes) → [batch,classes]
	if err != nil {
		return nil, err
	}
	sel, err := ex(backend.OpMul, nil, p, oneHot)
	if err != nil {
		return nil, err
	}
	pt, err := ex(backend.OpSum, lastAxis, sel) // p_t = softmax prob of the true class, [batch,1]
	if err != nil {
		return nil, err
	}
	logpt, err := ex(backend.OpLog, nil, pt)
	if err != nil {
		return nil, err
	}
	// per-example modulated log-prob t = (1−p_t)^γ · log p_t (γ=0 ⇒ just log p_t).
	t := logpt
	if gamma != 0 {
		oneMinus, err := ex(backend.OpSub, nil, scalarTensor(pt.Dtype(), 1), pt) // 1 − p_t
		if err != nil {
			return nil, err
		}
		logOneMinus, err := ex(backend.OpLog, nil, oneMinus)
		if err != nil {
			return nil, err
		}
		scaled, err := ex(backend.OpMul, nil, logOneMinus, scalarTensor(pt.Dtype(), gamma)) // γ·log(1−p_t)
		if err != nil {
			return nil, err
		}
		mod, err := ex(backend.OpExp, nil, scaled) // (1−p_t)^γ
		if err != nil {
			return nil, err
		}
		if t, err = ex(backend.OpMul, nil, mod, logpt); err != nil {
			return nil, err
		}
	}
	meanT, err := ex(backend.OpMean, nil, t) // scalar mean over the batch
	if err != nil {
		return nil, err
	}
	// loss = −α · mean(t), scaled without promoting the rank-0 scalar (AXPY + rank-0 zero).
	return ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: -alpha}, meanT, tensor.New(meanT.Dtype(), meanT.Shape()))
}

// SigmoidFocalLoss is the BINARY (sigmoid) focal loss — the paper's original form (Lin et al. 2018,
// arXiv:1708.02002, Eq. 4/5; the torchvision sigmoid_focal_loss), the per-element sibling of the
// multi-class FocalLoss above. Each logit is an independent binary classification: with p = σ(x)
// the model's P(y=1), p_t = p if y=1 else 1−p, and
//
//	FL = −α_t·(1−p_t)^γ·log(p_t)
//
// down-weights well-classified elements (p_t→1 ⇒ factor→0). γ=0 recovers (α-weighted) binary
// cross-entropy. α balances the classes: α_t = α for positives and 1−α for negatives; pass α<0 to
// disable α-weighting (α_t=1), as torchvision does. logits and targets (0/1) are the same shape;
// the loss is the mean and is differentiable w.r.t. the logits. Computed stably in softplus space:
// with z = (1−2y)·x, log p_t = −softplus(z) and (1−p_t)^γ = exp(−γ·softplus(−z)), so
// FL = α_t·softplus(z)·exp(−γ·softplus(−z)) — no σ/log that could under/overflow (paper Eq.4/5,
// verified under §R228).
func SigmoidFocalLoss(ctx *backend.Context, logits, targets *tensor.Tensor, gamma, alpha float64) (*tensor.Tensor, error) {
	if !logits.Shape().Equal(targets.Shape()) {
		return nil, fmt.Errorf("nn: SigmoidFocalLoss needs equal-shaped logits/targets, got %v and %v", logits.Shape(), targets.Shape())
	}
	// per-element constants from the (non-differentiable) targets: sign s=1−2y and weight α_t.
	sign := tensor.New(logits.Dtype(), logits.Shape())
	alphaT := tensor.New(logits.Dtype(), logits.Shape())
	for i := range targets.Numel() {
		c := tensor.Unravel(i, targets.Shape())
		y := targets.AtF64(c...)
		sign.SetF64(1-2*y, c...)
		switch {
		case alpha < 0:
			alphaT.SetF64(1, c...) // α-weighting disabled
		case y == 1:
			alphaT.SetF64(alpha, c...)
		default:
			alphaT.SetF64(1-alpha, c...)
		}
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	z, err := ex(backend.OpMul, nil, logits, sign) // z = (1−2y)·x
	if err != nil {
		return nil, err
	}
	spz, err := ex(backend.OpSoftplus, nil, z) // softplus(z) = −log p_t (the per-element BCE)
	if err != nil {
		return nil, err
	}
	term := spz
	if gamma != 0 { // (1−p_t)^γ = exp(−γ·softplus(−z))
		negZ, err := ex(backend.OpNeg, nil, z)
		if err != nil {
			return nil, err
		}
		spNegZ, err := ex(backend.OpSoftplus, nil, negZ)
		if err != nil {
			return nil, err
		}
		scaled, err := ex(backend.OpMul, nil, spNegZ, scalarTensor(z.Dtype(), -gamma))
		if err != nil {
			return nil, err
		}
		mod, err := ex(backend.OpExp, nil, scaled)
		if err != nil {
			return nil, err
		}
		if term, err = ex(backend.OpMul, nil, spz, mod); err != nil {
			return nil, err
		}
	}
	weighted, err := ex(backend.OpMul, nil, term, alphaT) // α_t·(1−p_t)^γ·(−log p_t)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMean, nil, weighted)
}
