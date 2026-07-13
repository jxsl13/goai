package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DINO self-distillation defaults (Caron et al. 2021, arXiv:2104.14294): student softmax
// temperature, teacher (sharpening) temperature, and center EMA momentum.
const (
	DINOStudentTemp    = 0.1  // τ_s
	DINOTeacherTemp    = 0.04 // τ_t (< τ_s ⇒ sharpened target)
	DINOCenterMomentum = 0.9  // m in the center EMA
)

// DINOLoss computes the DINO self-distillation loss (Caron, Touvron, Misra, Jégou, Mairal,
// Bojanowski & Joulin 2021, "Emerging Properties in Self-Supervised Vision Transformers",
// ICCV 2021, arXiv:2104.14294). DINO trains a student to match a momentum TEACHER (an EMA
// of the student, see nn.EMA) with NO labels, negatives, or clustering — distinct from
// InfoNCE (contrastive), Barlow/VICReg (statistics), and SwAV (Sinkhorn codes). Collapse
// is avoided by two opposing operations on the teacher: CENTERING (subtract a running mean
// C, so no single output dimension dominates) and SHARPENING (a small teacher temperature
// τ_t < τ_s, so the target stays peaked). The loss is the cross-entropy between the
// centered+sharpened teacher distribution and the student:
//
//	P_t = softmax((g_t − C)/τ_t),  P_s = softmax(g_s/τ_s),  L = −Σ_k P_t^(k) log P_s^(k)
//
// (mean over the batch). The teacher is a stop-gradient TARGET — gradients flow only to the
// student logits. studentLogits and teacherLogits are [B,K]; center is [1,K]; tauS, tauT
// are the temperatures. Differentiable w.r.t. studentLogits.
func DINOLoss(ctx *backend.Context, studentLogits, teacherLogits, center *tensor.Tensor, tauS, tauT float64) (*tensor.Tensor, error) {
	if studentLogits.Ndim() != 2 || !teacherLogits.Shape().Equal(studentLogits.Shape()) {
		return nil, fmt.Errorf("nn: DINOLoss needs equal-shaped [B,K] student/teacher logits, got %v and %v", studentLogits.Shape(), teacherLogits.Shape())
	}
	k := studentLogits.Shape()[1]
	if center.Ndim() != 2 || center.Shape()[0] != 1 || center.Shape()[1] != k {
		return nil, fmt.Errorf("nn: DINOLoss center must be [1,%d], got %v", k, center.Shape())
	}
	if tauS <= 0 || tauT <= 0 {
		return nil, fmt.Errorf("nn: DINOLoss temperatures must be > 0, got τ_s=%g τ_t=%g", tauS, tauT)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// centered + sharpened teacher target: P_t = softmax((g_t − C)/τ_t), stop-gradient.
	centered, err := ex(backend.OpSub, nil, teacherLogits, center) // [B,K] − [1,K]
	if err != nil {
		return nil, err
	}
	sharp, err := ex(backend.OpMul, nil, centered, scalarTensor(teacherLogits.Dtype(), 1/tauT))
	if err != nil {
		return nil, err
	}
	pt, err := ex(backend.OpSoftmax, nil, sharp)
	if err != nil {
		return nil, err
	}
	ptSG, err := ex(backend.OpStopGradient, nil, pt) // teacher is a fixed target
	if err != nil {
		return nil, err
	}
	// cross-entropy H(P_t, P_s) = mean_b(−Σ_k P_t·log softmax(g_s/τ_s)) — reuse swavSoftCE.
	return swavSoftCE(ctx, studentLogits, ptSG, tauS)
}

// DINOCenterUpdate advances the DINO centering vector by one EMA step:
// C ← m·C + (1−m)·mean_batch(teacherLogits) (§3.1 Eq. 4). center is [1,K], teacherLogits [B,K],
// momentum m ∈ [0,1); returns the updated center [1,K]. This is a running statistic (not
// differentiable), applied between batches.
func DINOCenterUpdate(center, teacherLogits *tensor.Tensor, momentum float64) (*tensor.Tensor, error) {
	if center.Ndim() != 2 || center.Shape()[0] != 1 {
		return nil, fmt.Errorf("nn: DINOCenterUpdate center must be [1,K], got %v", center.Shape())
	}
	if teacherLogits.Ndim() != 2 || teacherLogits.Shape()[1] != center.Shape()[1] {
		return nil, fmt.Errorf("nn: DINOCenterUpdate teacherLogits must be [B,%d], got %v", center.Shape()[1], teacherLogits.Shape())
	}
	b, k := teacherLogits.Shape()[0], teacherLogits.Shape()[1]
	out := tensor.New(center.Dtype(), tensor.Shape{1, k})
	for j := range k {
		var mean float64
		for i := range b {
			mean += teacherLogits.AtF64(i, j)
		}
		mean /= float64(b)
		out.SetF64(momentum*center.AtF64(0, j)+(1-momentum)*mean, 0, j)
	}
	return out, nil
}
