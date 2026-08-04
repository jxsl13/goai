package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// PlackettLuceLoss is the listwise ranking loss (ListMLE; Xia, Liu, Wang, Zhang & Li 2008;
// the Plackett-Luce model, Plackett 1975 / Luce 1959) for learning a reward/score model from
// full K-way rankings — the listwise generalization of the pairwise Bradley-Terry loss
// (nn.BradleyTerryLoss). Under Plackett-Luce the probability of a ranking is a product of
// sequential softmax choices — at each rank the best remaining item is drawn by a softmax
// over the still-unranked items — so the negative log-likelihood of a ranking with scores
// s (given in DESCENDING preference order, s[0] the most preferred) is
//
//	L = Σ_{k=0}^{K-2} [ logsumexp(s[k:]) − s[k] ]
//
// i.e. at each rank k the current best s[k] competes against the suffix s[k:]. It reduces
// exactly to the Bradley-Terry loss for K=2, and equals log(K!) for uniform scores.
//
// rewards is [batch, K] with each row a ranking's scores in descending preference order; the
// loss is the batch-mean and is differentiable w.r.t. the rewards (hence the reward model).
func PlackettLuceLoss(ctx *backend.Context, rewards *tensor.Tensor) (*tensor.Tensor, error) {
	if rewards.Ndim() != 2 {
		return nil, fmt.Errorf("nn: PlackettLuceLoss wants rank-2 [batch,K] rewards, got %v", rewards.Shape())
	}
	b, k := rewards.Shape()[0], rewards.Shape()[1]
	if k < 2 {
		return nil, fmt.Errorf("nn: PlackettLuceLoss needs K≥2 ranked items, got %d", k)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	lastAxis := backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}
	neg := math.Inf(-1)
	var total *tensor.Tensor
	//perfscan:ignore PS4011 small [batch,K] listwise loss, dominated by reward-model fwd/bwd
	for rank := 0; rank < k-1; rank++ {
		// suffix logsumexp: mask columns < rank to −∞ (so exp→0), logsumexp over the row.
		//perfscan:ignore PS6016 resource-only tiny maskRow alloc-in-loop
		maskRow := tensor.New(rewards.Dtype(), tensor.Shape{1, k})
		for j := 0; j < rank; j++ {
			maskRow.SetF64(neg, 0, j)
		}
		masked, err := ex(backend.OpAdd, nil, rewards, maskRow)
		if err != nil {
			return nil, err
		}
		mx, err := ex(backend.OpMax, lastAxis, masked)
		if err != nil {
			return nil, err
		}
		shifted, err := ex(backend.OpSub, nil, masked, mx)
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
		lg, err := ex(backend.OpLog, nil, sumexp)
		if err != nil {
			return nil, err
		}
		lse, err := ex(backend.OpAdd, nil, lg, mx) // logsumexp(s[rank:]) [batch,1]
		if err != nil {
			return nil, err
		}
		// s[:,rank] via a one-hot column selector (avoids a slice op).
		//perfscan:ignore PS6016 resource-only tiny oneHot alloc-in-loop
		oneHot := tensor.New(rewards.Dtype(), tensor.Shape{1, k})
		oneHot.SetF64(1, 0, rank)
		sel, err := ex(backend.OpMul, nil, rewards, oneHot)
		if err != nil {
			return nil, err
		}
		sk, err := ex(backend.OpSum, lastAxis, sel) // [batch,1]
		if err != nil {
			return nil, err
		}
		term, err := ex(backend.OpSub, nil, lse, sk) // logsumexp(s[rank:]) − s[rank]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016 resource-only, tiny scalar-reduce alloc
		s, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, term) // scalar
		if err != nil {
			return nil, err
		}
		if total == nil {
			total = s
		} else if total, err = ex(backend.OpAdd, nil, total, s); err != nil {
			return nil, err
		}
	}
	// batch mean = total/b, scaled without promoting the scalar's rank (AXPY + rank-0 zero).
	return ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: 1 / float64(b)}, total, tensor.New(total.Dtype(), total.Shape()))
}
