package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func rdropExec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// SymmetricKL is the R-Drop regularizer (Wu et al. 2021, arXiv:2106.14448, §R135) — the bidirectional
// KL divergence between the softmax distributions of two logit tensors over their last axis:
//
//	L_KL = ½·( KL(P1 ‖ P2) + KL(P2 ‖ P1) ) = ½·Σ (p−q)·(log p − log q) ,   p=softmax(l1), q=softmax(l2)
//
// returned as the mean over rows. Unlike knowledge distillation (where the teacher is frozen), the
// gradient flows through BOTH distributions — this is composed from the differentiable softmax, log,
// subtract, multiply and sum ops. l1 and l2 must share a shape (rank ≥ 1, last axis = classes). This
// is the core of R-Drop: run the same input through the network twice with independent dropout, then
// add α·SymmetricKL(logits1, logits2) to the task loss (see RDropLoss).
func SymmetricKL(ctx *backend.Context, l1, l2 *tensor.Tensor) (*tensor.Tensor, error) {
	if l1.Ndim() < 1 || !l1.Shape().Equal(l2.Shape()) {
		return nil, fmt.Errorf("nn: SymmetricKL needs equal-shaped rank≥1 logits, got %v and %v", l1.Shape(), l2.Shape())
	}
	p, err := rdropExec(ctx, backend.OpSoftmax, nil, l1)
	if err != nil {
		return nil, err
	}
	q, err := rdropExec(ctx, backend.OpSoftmax, nil, l2)
	if err != nil {
		return nil, err
	}
	lp, err := rdropExec(ctx, backend.OpLog, nil, p)
	if err != nil {
		return nil, err
	}
	lq, err := rdropExec(ctx, backend.OpLog, nil, q)
	if err != nil {
		return nil, err
	}
	dp, err := rdropExec(ctx, backend.OpSub, nil, p, q) // p − q
	if err != nil {
		return nil, err
	}
	dl, err := rdropExec(ctx, backend.OpSub, nil, lp, lq) // log p − log q
	if err != nil {
		return nil, err
	}
	prod, err := rdropExec(ctx, backend.OpMul, nil, dp, dl)
	if err != nil {
		return nil, err
	}
	total, err := rdropExec(ctx, backend.OpSum, nil, prod) // Σ over all elements
	if err != nil {
		return nil, err
	}
	d := l1.Shape()[l1.Ndim()-1]
	rows := l1.Numel() / d
	// ½·total / rows  =  mean over rows of the per-row symmetric KL
	return rdropExec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: 0.5 / float64(rows)}, total, tensor.New(l1.Dtype(), tensor.Shape{}))
}

// RDropLoss is the full R-Drop training loss (Wu et al. 2021) for two dropout forward passes over the
// same input: logits1, logits2 ∈ [batch, classes] with integer targets ∈ [batch]:
//
//	L = ½·( CrossEntropy(logits1, y) + CrossEntropy(logits2, y) ) + α·SymmetricKL(logits1, logits2)
//
// The CE term averages the two passes' cross-entropies (the official repo's form; the paper's Eq. 3
// sums them — equivalent up to scaling), and α weights the symmetric-KL consistency regularizer
// (paper α: ~5 for NMT, ~1 for language modeling, smaller for classification). R-Drop adds no
// parameters and no architecture change — only this loss on top of the existing dropout.
func RDropLoss(ctx *backend.Context, logits1, logits2, targets *tensor.Tensor, alpha float64) (*tensor.Tensor, error) {
	ce1, err := CrossEntropy(ctx, logits1, targets)
	if err != nil {
		return nil, err
	}
	ce2, err := CrossEntropy(ctx, logits2, targets)
	if err != nil {
		return nil, err
	}
	ceSum, err := rdropExec(ctx, backend.OpAdd, nil, ce1, ce2)
	if err != nil {
		return nil, err
	}
	ceAvg, err := rdropExec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: 0.5}, ceSum, tensor.New(logits1.Dtype(), tensor.Shape{}))
	if err != nil {
		return nil, err
	}
	kl, err := SymmetricKL(ctx, logits1, logits2)
	if err != nil {
		return nil, err
	}
	// α·kl + ceAvg
	return rdropExec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: alpha}, kl, ceAvg)
}
