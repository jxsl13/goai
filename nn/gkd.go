package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GKDLoss is the generalized Jensen-Shannon divergence used by GKD (Agarwal, Vieillard,
// Zhou, Stanczyk, Ramos, Geist & Bachem 2024, "On-Policy Distillation of Language Models",
// arXiv:2306.13649, ICLR'24) for knowledge distillation. With the teacher distribution
// P = softmax(teacherLogits), the student Q = softmax(studentLogits), and the mixture
// M = β·P + (1−β)·Q over the last (vocab) axis:
//
//	JSD^β(P‖Q) = β·KL(P‖M) + (1−β)·KL(Q‖M)          (β ∈ (0,1))
//
// averaged over rows. Unlike KL, the generalized JSD is bounded even for disjoint
// supports, and it interpolates the distillation objective: β→0 behaves like the forward
// KL(P‖Q) (teacher‖student) and β→1 like the reverse KL(Q‖P); β=0.5 is the standard
// symmetric Jensen-Shannon divergence. The degenerate endpoints are taken literally —
// β=0 is the forward KL(teacher‖student) and β=1 the reverse KL(student‖teacher) — so the
// loss is continuous and useful across the whole [0,1] range. It is differentiable w.r.t.
// the student logits (the distillation target). β must be in [0,1].
func GKDLoss(ctx *backend.Context, studentLogits, teacherLogits *tensor.Tensor, beta float64) (*tensor.Tensor, error) {
	if studentLogits.Ndim() < 1 || !studentLogits.Shape().Equal(teacherLogits.Shape()) {
		return nil, fmt.Errorf("nn: GKDLoss needs equal-shaped rank≥1 logits, got %v and %v", studentLogits.Shape(), teacherLogits.Shape())
	}
	if beta < 0 || beta > 1 {
		return nil, fmt.Errorf("nn: GKDLoss beta must be in [0,1], got %g", beta)
	}
	p, err := rdropExec(ctx, backend.OpSoftmax, nil, teacherLogits) // P = teacher
	if err != nil {
		return nil, err
	}
	q, err := rdropExec(ctx, backend.OpSoftmax, nil, studentLogits) // Q = student
	if err != nil {
		return nil, err
	}
	logP, err := rdropExec(ctx, backend.OpLog, nil, p)
	if err != nil {
		return nil, err
	}
	logQ, err := rdropExec(ctx, backend.OpLog, nil, q)
	if err != nil {
		return nil, err
	}

	var terms *tensor.Tensor
	switch beta {
	case 0: // forward KL(P‖Q) = Σ P·(log P − log Q)
		terms, err = klTerms(ctx, p, logP, logQ)
	case 1: // reverse KL(Q‖P) = Σ Q·(log Q − log P)
		terms, err = klTerms(ctx, q, logQ, logP)
	default:
		// M = β·P + (1−β)·Q = Q + β·(P−Q)
		var pmq, m, logM, klP, klQ, scaledQ *tensor.Tensor
		if pmq, err = rdropExec(ctx, backend.OpSub, nil, p, q); err != nil {
			return nil, err
		}
		if m, err = rdropExec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: beta}, pmq, q); err != nil {
			return nil, err
		}
		if logM, err = rdropExec(ctx, backend.OpLog, nil, m); err != nil {
			return nil, err
		}
		if klP, err = klTerms(ctx, p, logP, logM); err != nil { // P·log(P/M)
			return nil, err
		}
		if klQ, err = klTerms(ctx, q, logQ, logM); err != nil { // Q·log(Q/M)
			return nil, err
		}
		// terms = β·klP + (1−β)·klQ
		if scaledQ, err = rdropExec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: 1 - beta}, klQ, tensor.New(klQ.Dtype(), klQ.Shape())); err != nil {
			return nil, err
		}
		terms, err = rdropExec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: beta}, klP, scaledQ)
	}
	if err != nil {
		return nil, err
	}

	total, err := rdropExec(ctx, backend.OpSum, nil, terms) // Σ over all elements
	if err != nil {
		return nil, err
	}
	d := studentLogits.Shape()[studentLogits.Ndim()-1]
	rows := studentLogits.Numel() / d
	// mean over rows
	return rdropExec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: 1 / float64(rows)}, total, tensor.New(studentLogits.Dtype(), tensor.Shape{}))
}

// klTerms returns the elementwise KL contribution a·(logA − logB) (summed later).
func klTerms(ctx *backend.Context, a, logA, logB *tensor.Tensor) (*tensor.Tensor, error) {
	diff, err := rdropExec(ctx, backend.OpSub, nil, logA, logB)
	if err != nil {
		return nil, err
	}
	return rdropExec(ctx, backend.OpMul, nil, a, diff)
}
