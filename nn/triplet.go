package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TripletLoss is the triplet margin loss (Schroff, Kalenichenko & Philbin 2015, "FaceNet: A
// Unified Embedding for Face Recognition and Clustering", CVPR, arXiv:1503.03832, §3.1 — the loss
// is the paper's Eq. 3; Eq. 1 is its margin constraint). It
// trains an embedding so that, for each triplet, the anchor is closer to a POSITIVE (same class)
// than to a NEGATIVE (different class) by at least a margin — the margin-based metric-learning
// counterpart of the contrastive/redundancy-reduction objectives here (InfoNCE, Barlow Twins,
// VICReg, SwAV, DINO, SimSiam). With squared Euclidean distances,
//
//	L = mean [ ‖a − p‖² − ‖a − n‖² + α ]_+
//
// where [·]_+ = max(0, ·). The loss is zero exactly when every negative is at least α farther from
// its anchor than the positive (‖a−n‖² ≥ ‖a−p‖² + α); otherwise it pulls the positive in and pushes
// the negative out. FaceNet L2-normalizes the embeddings onto the unit hypersphere first — do that
// upstream (as with the other embedding losses); this operates on the given vectors. anchor,
// positive and negative are [batch, dim]; the loss is the batch mean and is differentiable w.r.t.
// all three. Paper default margin α = 0.2.
func TripletLoss(ctx *backend.Context, anchor, positive, negative *tensor.Tensor, margin float64) (*tensor.Tensor, error) {
	if !anchor.Shape().Equal(positive.Shape()) || !anchor.Shape().Equal(negative.Shape()) {
		return nil, fmt.Errorf("nn: TripletLoss needs equal-shaped [batch,dim] anchor/positive/negative, got %v, %v, %v", anchor.Shape(), positive.Shape(), negative.Shape())
	}
	if anchor.Ndim() != 2 {
		return nil, fmt.Errorf("nn: TripletLoss wants rank-2 [batch,dim] inputs, got %v", anchor.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	lastAxis := backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}
	// squared Euclidean distance ‖u−v‖² per row → [batch,1].
	sqDist := func(u, v *tensor.Tensor) (*tensor.Tensor, error) {
		d, err := ex(backend.OpSub, nil, u, v)
		if err != nil {
			return nil, err
		}
		sq, err := ex(backend.OpMul, nil, d, d)
		if err != nil {
			return nil, err
		}
		return ex(backend.OpSum, lastAxis, sq)
	}
	dPos, err := sqDist(anchor, positive)
	if err != nil {
		return nil, err
	}
	dNeg, err := sqDist(anchor, negative)
	if err != nil {
		return nil, err
	}
	gap, err := ex(backend.OpSub, nil, dPos, dNeg) // ‖a−p‖² − ‖a−n‖²
	if err != nil {
		return nil, err
	}
	z, err := ex(backend.OpAdd, nil, gap, scalarTensor(gap.Dtype(), margin)) // + α
	if err != nil {
		return nil, err
	}
	hinge, err := ex(backend.OpReLU, nil, z) // max(0, ·)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMean, nil, hinge)
}
