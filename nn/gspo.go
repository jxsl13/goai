package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GSPOOption configures GSPOLoss (functional options, §C12).
type GSPOOption func(*gspoConfig)

type gspoConfig struct{ epsilon float64 }

// WithGSPOClipEpsilon sets the sequence-level clip range ε for the GSPO policy ratio.
//
// In plain terms: like PPO, GSPO refuses to trust a policy update that changed the generation
// probability too much in one step, and clips it back; ε is how much change is tolerated. But
// GSPO clips a LENGTH-NORMALIZED whole-sequence ratio (not per-token), and those ratios sit
// very close to 1, so its ε is far smaller than PPO/GRPO's. Boundary behavior — ε→0 clips
// almost every update (over-conservative, slow learning); too large lets destabilizing jumps
// through. A per-token ε like PPO's 0.2 would be meaningless here.
//
// Default 3e-4 (research-grounded: the GSPO paper's regime — Zheng et al. 2025, Qwen3's RL
// objective, §T549 — orders of magnitude tighter than PPO/GRPO's 0.2 because the sequence
// ratios concentrate near 1).
func WithGSPOClipEpsilon(eps float64) GSPOOption { return func(c *gspoConfig) { c.epsilon = eps } }

// GSPOLoss is the Group Sequence Policy Optimization loss (Zheng et al. 2025,
// arXiv:2507.18071 — the critic-free RL objective behind Qwen3, §T549). It is
// GRPO with the token-level importance ratios replaced by ONE length-normalized
// SEQUENCE likelihood ratio per response,
//
//	s_i  = ( πθ(y_i|x) / π_old(y_i|x) )^(1/|y_i|)
//	loss = −(1/G) Σ_i min(s_i·Â_i, clip(s_i, 1−ε, 1+ε)·Â_i)
//
// which matches the sequence-level reward attribution (the whole response earned
// the reward, not each token independently) and removes GRPO's high-variance
// per-token ratio noise on long responses. logpNew/logpOld are the FLAT [Σ|y_i|]
// concatenations of the G responses' per-token log-probs (TokenLogProbs);
// lengths segments them; advantage is [G] (GroupAdvantage). No KL term — the
// paper drops it. Gradients flow only into logpNew. With every length 1 the loss
// equals GRPOLoss with β=0 exactly.
func GSPOLoss(ctx *backend.Context, logpNew, logpOld, advantage *tensor.Tensor, lengths []int, opts ...GSPOOption) (*tensor.Tensor, error) {
	if len(lengths) == 0 {
		return nil, fmt.Errorf("nn: GSPOLoss needs per-sequence lengths")
	}
	cfg := gspoConfig{}
	for _, o := range opts {
		o(&cfg)
	}
	out, err := backend.Execute(ctx, backend.OpGSPO,
		[]*tensor.Tensor{logpNew, logpOld, advantage},
		backend.GSPOAttrs{Epsilon: cfg.epsilon, Lengths: lengths})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
