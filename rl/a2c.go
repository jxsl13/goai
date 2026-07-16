package rl

import (
	"math/rand/v2"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// --- A2C (Advantage Actor-Critic, the synchronous variant of the A3C algorithm,
// Mnih et al. 2016, arXiv:1602.01783) ---

// A2C is the simpler actor-critic baseline PPO builds on: the same policy πθ(a|s)
// and value Vφ(s) networks and the same [GAE] advantages, but a SINGLE
// unclipped policy-gradient step per rollout — loss = −mean(logπθ(a|s)·Â) − c2·H —
// instead of PPO's K clipped epochs. It is exactly PPO with one epoch and clipping
// disabled; the value baseline Â (advantage) is what reduces the gradient variance
// of plain REINFORCE. Deterministic given its seed. Configure the GAE λ, value and
// entropy coefficients, and rollout length with the WithA2C* options.
type A2C struct {
	Actor  *nn.Sequential // policy network: obs → hidden → NumActions logits (softmax policy)
	Critic *nn.Sequential // value network: obs → hidden → 1 scalar state value Vφ(s)
	Gamma  float64        // discount factor γ used by the GAE advantage/return computation
	Lambda float64        // GAE trade-off λ (0 = one-step TD, 1 = Monte-Carlo); default 0.95

	// ValueCoef scales the critic's MSE loss (c1). Default 0.5; 0 freezes the critic.
	ValueCoef float64
	// EntropyCoef weights the entropy bonus (c2) added to the policy objective.
	// Default 0.01; 0 disables it.
	EntropyCoef float64
	// RolloutSteps is the minimum number of transitions gathered per Update before
	// the single actor step and value step run; collection finishes the running
	// episode. Default 256.
	RolloutSteps int

	rng        *rand.Rand
	aOpt, cOpt nn.Optimizer
}

// A2COption configures an [A2C] agent at construction (functional-options pattern).
type A2COption func(*A2C)

// WithA2CLambda sets the GAE trade-off λ∈[0,1] (default 0.95). Values outside
// [0,1] are ignored.
func WithA2CLambda(lambda float64) A2COption {
	return func(a *A2C) {
		if lambda >= 0 && lambda <= 1 {
			a.Lambda = lambda
		}
	}
}

// WithA2CValueCoef sets the value-loss coefficient c1 (default 0.5). Negative
// values are ignored; 0 freezes the critic.
func WithA2CValueCoef(c float64) A2COption {
	return func(a *A2C) {
		if c >= 0 {
			a.ValueCoef = c
		}
	}
}

// WithA2CEntropyCoef sets the entropy-bonus coefficient c2 (default 0.01). Negative
// values are ignored; 0 removes the entropy term.
func WithA2CEntropyCoef(c float64) A2COption {
	return func(a *A2C) {
		if c >= 0 {
			a.EntropyCoef = c
		}
	}
}

// WithA2CRolloutSteps sets the minimum number of transitions collected per Update
// (default 256). Values ≤0 are ignored.
func WithA2CRolloutSteps(n int) A2COption {
	return func(a *A2C) {
		if n > 0 {
			a.RolloutSteps = n
		}
	}
}

// NewA2C builds an A2C agent for env with a hidden layer of the given width, Adam
// learning rate lr and discount γ, seeded deterministically. Defaults: λ=0.95,
// c1=0.5, c2=0.01, 256-step rollouts.
func NewA2C(env Env, hidden int, lr, gamma float64, seed uint64, opts ...A2COption) *A2C {
	actor := policyNet(env.ObsDim(), hidden, env.NumActions(), seed)
	critic := nn.NewSequential(
		nn.NewLinear(tensor.F64, env.ObsDim(), hidden, seed+2),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, hidden, 1, seed+3),
	)
	a := &A2C{
		Actor: actor, Critic: critic,
		Gamma: gamma, Lambda: 0.95,
		ValueCoef: 0.5, EntropyCoef: 0.01, RolloutSteps: 256,
		rng:  rand.New(rand.NewPCG(seed, 0x9b0)),
		aOpt: nn.NewAdam(actor.Params(), lr),
		cOpt: nn.NewAdam(critic.Params(), lr),
	}
	for _, o := range opts {
		o(a)
	}
	return a
}

// Update collects one on-policy rollout, computes GAE advantages/returns from the
// critic's values, and applies a single unclipped actor step and value-regression
// step. It returns the average completed-episode return in the rollout.
func (a *A2C) Update(env Env) (float64, error) {
	ro, err := rlRollout(env, a.Actor, a.Critic, a.rng, a.RolloutSteps)
	if err != nil {
		return 0, err
	}
	adv, rets := GAE(ro.rewards, ro.values, ro.dones, 0, true, a.Gamma, a.Lambda)
	advT := rlNormAdv(adv)

	tape := autograd.NewTape()
	ctx := tape.Context()
	loss, err := a2cPolicyLoss(ctx, a.Actor, ro.states, ro.actions, advT, a.EntropyCoef)
	if err != nil {
		return 0, err
	}
	if err := tape.Backward(loss); err != nil {
		return 0, err
	}
	if err := a.aOpt.Step(tape.Grad); err != nil {
		return 0, err
	}
	if err := rlValueFit(a.Critic, a.cOpt, ro.states, rets, a.ValueCoef, 1); err != nil {
		return 0, err
	}
	return ro.meanReturn(), nil
}

// Value returns the critic's estimate Vφ(obs) for a single observation.
func (a *A2C) Value(obs []float64) (float64, error) {
	v, err := forward(backend.NewContext(), a.Critic, [][]float64{obs})
	if err != nil {
		return 0, err
	}
	return v.AtF64(0, 0), nil
}

// ActionProbs returns the current softmax policy πθ(·|obs) for one observation —
// the post-training policy to inspect which action the agent favors.
func (a *A2C) ActionProbs(obs []float64) ([]float64, error) {
	logits, err := forward(backend.NewContext(), a.Actor, [][]float64{obs})
	if err != nil {
		return nil, err
	}
	probs, err := backend.Execute(backend.NewContext(), backend.OpSoftmax, []*tensor.Tensor{logits}, nil)
	if err != nil {
		return nil, err
	}
	out := make([]float64, probs[0].Shape()[1])
	for i := range out {
		out[i] = probs[0].AtF64(0, i)
	}
	return out, nil
}

// a2cPolicyLoss builds the differentiable vanilla actor-critic policy objective
// −mean(logπθ(a|s)·Â) − c2·H for one batch. It is PPO's clipped surrogate with the
// clip removed (ratio ρ replaced by 1 at collection-time params, where PPO's
// first-epoch gradient coincides). Shared with the collapse test.
func a2cPolicyLoss(ctx *backend.Context, actor *nn.Sequential, states [][]float64, actions []int, advT *tensor.Tensor, entropyCoef float64) (*tensor.Tensor, error) {
	logits, err := forward(ctx, actor, states)
	if err != nil {
		return nil, err
	}
	lp, err := nn.TokenLogProbs(ctx, logits, actions)
	if err != nil {
		return nil, err
	}
	weighted, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{lp, advT}, nil)
	if err != nil {
		return nil, err
	}
	mean, err := backend.Execute(ctx, backend.OpMean, weighted, nil)
	if err != nil {
		return nil, err
	}
	loss, err := backend.Execute(ctx, backend.OpNeg, mean, nil)
	if err != nil {
		return nil, err
	}
	out := loss[0]
	if entropyCoef != 0 {
		negH, err := rlNegEntropy(ctx, logits)
		if err != nil {
			return nil, err
		}
		term, err := rlScale(ctx, negH, entropyCoef)
		if err != nil {
			return nil, err
		}
		added, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{out, term}, nil)
		if err != nil {
			return nil, err
		}
		out = added[0]
	}
	return out, nil
}
