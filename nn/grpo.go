package nn

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GroupAdvantage computes the GRPO group-relative advantage (Shao et al. 2024,
// DeepSeekMath, arXiv:2402.03300): for the G sampled outputs of one prompt with
// scalar rewards r₁..r_G,
//
//	Âᵢ = (rᵢ − mean(r)) / (std(r) + eps)
//
// where std is the POPULATION standard deviation (matching the verl reference)
// and eps = 1e-4 guards the all-equal-reward case (matching TRL). This is
// "outcome supervision": the returned Âᵢ is then broadcast to every token of
// output i. It replaces PPO's learned value baseline — GRPO needs no critic
// network, which is why it is cheaper to run. Rewards are rollout-time constants,
// so this is a plain (non-differentiable) reduction. A single reward, or all-equal
// rewards, yields all-zero advantages (no learning signal from that group).
func GroupAdvantage(rewards []float64) []float64 {
	const eps = 1e-4
	n := len(rewards)
	adv := make([]float64, n)
	if n == 0 {
		return adv
	}
	var mean float64
	for _, r := range rewards {
		mean += r
	}
	mean /= float64(n)
	var variance float64
	for _, r := range rewards {
		d := r - mean
		variance += d * d
	}
	variance /= float64(n) // population (ddof=0)
	std := math.Sqrt(variance)
	for i, r := range rewards {
		adv[i] = (r - mean) / (std + eps)
	}
	return adv
}

// DrGRPOAdvantage computes the Dr. GRPO ("GRPO Done Right") group advantage (Liu et
// al. 2025, "Understanding R1-Zero-Like Training", arXiv:2503.20783 §3.1, §R95): the
// mean-centered reward WITHOUT GRPO's standard-deviation normalization,
//
//	Âᵢ = rᵢ − mean(r)
//
// Dropping the /std (compare [GroupAdvantage]) removes GRPO's question-level DIFFICULTY
// BIAS: dividing by the group's reward std rescales every group to unit variance, so
// low-variance prompts (near all-correct or all-wrong) get their tiny reward
// differences amplified to the same gradient magnitude as high-variance ones. Dr. GRPO
// instead preserves the raw reward scale, weighting each prompt by how much its outputs
// actually differ. Paired with GRPOLoss — whose global token mean is already a constant
// (non-per-response) normalizer, so it does not carry GRPO's response-length bias — and
// β=0 (WithKLBeta(0), as the paper uses), this gives the unbiased Dr. GRPO objective
// that prevents runaway response-length inflation. Rewards are rollout-time constants;
// Σ Âᵢ = 0; a single reward yields a zero advantage.
func DrGRPOAdvantage(rewards []float64) []float64 {
	n := len(rewards)
	adv := make([]float64, n)
	if n == 0 {
		return adv
	}
	var mean float64
	for _, r := range rewards {
		mean += r
	}
	mean /= float64(n)
	for i, r := range rewards {
		adv[i] = r - mean
	}
	return adv
}

// grpoConfig holds the GRPO hyperparameters (functional options, §C12).
type grpoConfig struct {
	epsilon float64 // PPO clip range
	beta    float64 // KL-to-reference penalty weight
}

// GRPOOption configures GRPOLoss.
type GRPOOption func(*grpoConfig)

// WithClipEpsilon sets the PPO-style probability-ratio clip range ε.
//
// In plain terms: GRPO won't trust a policy update that changed a token's probability by more
// than a factor of roughly 1±ε in one step — it clips the update back to stay stable. ε is how
// much change is allowed. Boundary behavior — ε→0 clips almost everything (over-conservative,
// slow); large ε removes the guardrail and lets destabilizing jumps through. This is a
// PER-TOKEN ratio (contrast GSPO's tiny sequence-level ε, which lives on a different scale).
//
// Default 0.2 (research-grounded: the PPO/GRPO reference clip range, Shao et al. 2024 §R95,
// inherited from PPO — Schulman et al. 2017).
func WithClipEpsilon(eps float64) GRPOOption { return func(c *grpoConfig) { c.epsilon = eps } }

// WithKLBeta sets β, the weight of the KL-to-reference penalty (the k3 estimator) that keeps
// the trained policy from drifting too far from the reference model.
//
// In plain terms: a leash — β controls how strongly the policy is pulled back toward the
// original model, trading reward-chasing against staying sensible. Boundary behavior — larger β
// keeps the policy very close to the reference (safe but limits how much it can improve); as
// β→0 the leash loosens. SPECIAL VALUE: 0 disables the KL term entirely, as in some
// DeepSeek-R1-style runs that rely on clipping alone.
//
// Default 0.04 (research-grounded: the GRPO reference KL weight, Shao et al. 2024 §R95).
func WithKLBeta(beta float64) GRPOOption { return func(c *grpoConfig) { c.beta = beta } }

// GRPOLoss is the Group Relative Policy Optimization loss (Shao et al. 2024,
// DeepSeekMath) — the critic-free RL objective behind DeepSeek-R1. Given the
// current policy log-probs logπθ, the frozen rollout log-probs logπ_old, the
// reference-model log-probs logπ_ref, and the group-normalized advantages Â (all
// [batch], one entry per token), it returns
//
//	loss = −mean_i( min(rᵢ·Âᵢ, clip(rᵢ,1−ε,1+ε)·Âᵢ) − β·klᵢ )
//
// with rᵢ = exp(logπθᵢ − logπ_oldᵢ) and the unbiased Schulman k3 KL penalty
// klᵢ = exp(Δᵢ) − Δᵢ − 1, Δᵢ = logπ_refᵢ − logπθᵢ. It is the PPO clipped surrogate
// (nn.PPOClipLoss) plus an explicit KL-to-reference penalty, and needs no value
// network — the advantage comes from GroupAdvantage. logπ_old, logπ_ref and Â are
// rollout-time constants; gradients flow only into logπθ. Defaults ε=0.2, β=0.04.
func GRPOLoss(ctx *backend.Context, logpNew, logpOld, logpRef, advantage *tensor.Tensor, opts ...GRPOOption) (*tensor.Tensor, error) {
	cfg := grpoConfig{epsilon: 0.2, beta: 0.04}
	for _, o := range opts {
		o(&cfg)
	}
	out, err := backend.Execute(ctx, backend.OpGRPO,
		[]*tensor.Tensor{logpNew, logpOld, logpRef, advantage},
		backend.GRPOAttrs{Epsilon: cfg.epsilon, Beta: cfg.beta})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
