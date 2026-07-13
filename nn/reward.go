package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BradleyTerryLoss is the reward-model training loss of the classic RLHF pipeline
// (Christiano et al. 2017, arXiv:1706.03741; Stiennon et al. 2020; Ouyang et al. 2022
// InstructGPT; Touvron et al. 2023 Llama-2). A reward model r_θ(x,y) outputs a scalar score
// for a response; the Bradley-Terry preference model sets P(y_w ≻ y_l) = σ(r_w − r_l), and
// the reward model is trained by the negative log-likelihood of the observed preferences:
//
//	L = −mean log σ( r_chosen − r_rejected − margin )
//
// This is the pairwise logistic loss — the reward model this library's PPO/GRPO/RLOO
// optimize against, and the classic counterpart of the reward-model-FREE direct methods
// (DPO/IPO/KTO). The optional margin (Llama-2 §3.2.2) is SUBTRACTED inside the sigmoid so a
// stronger stated preference forces a larger reward gap (margin 0 = plain Bradley-Terry).
//
// rewardChosen and rewardRejected are [batch] scalar rewards from the reward model. The loss
// is differentiable w.r.t. both (hence the reward model), computed stably via
// −log σ(z) = softplus(−z): per pair softplus(r_rejected − r_chosen + margin).
func BradleyTerryLoss(ctx *backend.Context, rewardChosen, rewardRejected *tensor.Tensor, margin float64) (*tensor.Tensor, error) {
	if !rewardChosen.Shape().Equal(rewardRejected.Shape()) {
		return nil, fmt.Errorf("nn: BradleyTerryLoss needs equal-shaped chosen/rejected rewards, got %v and %v", rewardChosen.Shape(), rewardRejected.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// z = r_rejected − r_chosen (+ margin); per-pair loss = softplus(z) = −log σ(r_w−r_l−margin).
	diff, err := ex(backend.OpSub, nil, rewardRejected, rewardChosen)
	if err != nil {
		return nil, err
	}
	if margin != 0 {
		diff, err = ex(backend.OpAdd, nil, diff, scalarTensor(diff.Dtype(), margin))
		if err != nil {
			return nil, err
		}
	}
	sp, err := ex(backend.OpSoftplus, nil, diff)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMean, nil, sp)
}
