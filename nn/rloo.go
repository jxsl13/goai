package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// RLOOAdvantage computes the REINFORCE Leave-One-Out advantage (Ahmadian et al.
// 2024, "Back to Basics", arXiv:2402.14740; RLOO estimator, Kool et al. 2019). For
// the k sampled completions of one prompt with scalar rewards r₁..r_k, each
// sample's baseline is the MEAN of the OTHER k−1 rewards (leave-one-out), so
//
//	Âᵢ = rᵢ − (1/(k−1))·Σ_{j≠i} rⱼ = (k/(k−1))·(rᵢ − mean(r))
//
// This unbiased baseline needs no value network (unlike PPO) and no group std
// normalization (unlike GRPO's GroupAdvantage) — the "back to basics" result that
// plain REINFORCE with the LOO baseline matches PPO for RLHF. Rewards are
// rollout-time constants, so this is a plain (non-differentiable) reduction. With
// k≤1 there is no leave-one-out baseline, so all advantages are 0.
func RLOOAdvantage(rewards []float64) []float64 {
	k := len(rewards)
	adv := make([]float64, k)
	if k <= 1 {
		return adv
	}
	var sum float64
	for _, r := range rewards {
		sum += r
	}
	for i, r := range rewards {
		adv[i] = r - (sum-r)/float64(k-1) // rᵢ − mean of the other k−1
	}
	return adv
}

// RLOOLoss is the RLOO policy-gradient loss (Ahmadian et al. 2024): the REINFORCE
// objective with the leave-one-out baseline, no critic and no PPO clipping. Given
// the per-sample sequence log-probs logp[k] (a tape tensor gradients flow into) and
// the advantages Â[k] from RLOOAdvantage (rollout-time constants),
//
//	loss = −mean_i( Âᵢ · logp_i )
//
// so ∂loss/∂logpᵢ = −Âᵢ/k — pushing up the log-prob of above-average completions
// and down the below-average ones. It composes plain ops, so autograd differentiates
// it through logp with no fused kernel. Any KL-to-reference penalty is folded into
// the rewards before RLOOAdvantage (as in the paper), not added here.
func RLOOLoss(ctx *backend.Context, logp, advantage *tensor.Tensor) (*tensor.Tensor, error) {
	if !logp.Shape().Equal(advantage.Shape()) {
		return nil, fmt.Errorf("nn: RLOO logp shape %v != advantage %v", logp.Shape(), advantage.Shape())
	}
	//perfscan:ignore PS3038 resource-only slice-literal alloc per dispatch
	prod, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{advantage, logp}, nil)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 resource-only slice-literal alloc per dispatch
	m, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{prod[0]}, backend.ReduceAttrs{})
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 resource-only slice-literal alloc per dispatch
	neg, err := backend.Execute(ctx, backend.OpNeg, []*tensor.Tensor{m[0]}, nil)
	if err != nil {
		return nil, err
	}
	return neg[0], nil
}
