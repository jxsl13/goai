package rl

import (
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// --- PPO (Proximal Policy Optimization, Schulman et al. 2017, arXiv:1707.06347) ---

// PPO is the clipped-surrogate actor-critic agent — the workhorse of modern deep
// reinforcement learning (and, with token log-probs, of RLHF). A separate policy
// network πθ(a|s) (softmax over the discrete actions) and value network Vφ(s)
// are trained on-policy: each [Update] collects a rollout with the current policy,
// turns the critic's value estimates into [GAE] advantages, then takes K epochs of
// clipped policy-gradient steps and value-regression steps over that fixed batch.
//
// The clipped surrogate L^CLIP = mean min(ρÂ, clip(ρ,1−ε,1+ε)Â) with the
// probability ratio ρ = exp(logπθ − logπ_old) forms a trust region: once ρ leaves
// [1−ε,1+ε] in the pessimistic direction its gradient vanishes, so a single batch
// cannot push the policy too far. The policy objective additionally subtracts an
// entropy bonus (exploration) and the critic minimizes MSE(Vφ, GAE returns). The
// networks are composed from nn.* exactly like [Reinforce]/[DQN]; the agent is
// deterministic given its seed.
//
// Set ε with [WithPPOClip], K with [WithPPOEpochs], the GAE trade-off λ with
// [WithPPOLambda], the value/entropy weights with [WithPPOValueCoef]/
// [WithPPOEntropyCoef], and the rollout length with [WithPPORolloutSteps].
type PPO struct {
	Actor  *nn.Sequential // policy network: obs → hidden → NumActions logits (softmax policy)
	Critic *nn.Sequential // value network: obs → hidden → 1 scalar state value Vφ(s)
	Gamma  float64        // discount factor γ used by the GAE advantage/return computation
	Lambda float64        // GAE trade-off λ (0 = one-step TD, 1 = Monte-Carlo); default 0.95
	Clip   float64        // PPO clip ε: ratio trust region [1−ε, 1+ε]; default 0.2
	Epochs int            // number of gradient epochs K taken over each rollout batch; default 4

	// ValueCoef scales the critic's MSE loss (c1 in L − c1·valueLoss + c2·entropy).
	// Default 0.5. Since the critic has its own optimizer this simply rescales its
	// effective step; 0 freezes the critic.
	ValueCoef float64
	// EntropyCoef weights the entropy bonus (c2) added to the policy objective to
	// keep exploration alive. Default 0.01; 0 disables the bonus (pure clipped PG).
	EntropyCoef float64
	// RolloutSteps is the minimum number of environment transitions gathered per
	// Update before the policy/value steps run; collection always finishes the
	// current episode so the batch ends at a terminal state. Default 256.
	RolloutSteps int

	numActions int
	rng        *rand.Rand
	aOpt, cOpt nn.Optimizer
}

// PPOOption configures a [PPO] agent at construction (functional-options pattern).
type PPOOption func(*PPO)

// WithPPOClip sets the clip parameter ε (trust region [1−ε, 1+ε]). The PPO paper
// uses 0.2; ε≤0 is ignored (keeps the default). A very large ε disables clipping,
// collapsing the objective to vanilla actor-critic policy gradient (A2C).
func WithPPOClip(eps float64) PPOOption {
	return func(p *PPO) {
		if eps > 0 {
			p.Clip = eps
		}
	}
}

// WithPPOEpochs sets the number of gradient epochs K taken over each rollout batch
// (default 4). K≤0 is ignored. K=1 with no clip is the A2C update.
func WithPPOEpochs(k int) PPOOption {
	return func(p *PPO) {
		if k > 0 {
			p.Epochs = k
		}
	}
}

// WithPPOLambda sets the GAE trade-off λ∈[0,1] (default 0.95): λ=0 is the
// high-bias/low-variance one-step TD advantage, λ=1 the unbiased Monte-Carlo one.
// Values outside [0,1] are ignored.
func WithPPOLambda(lambda float64) PPOOption {
	return func(p *PPO) {
		if lambda >= 0 && lambda <= 1 {
			p.Lambda = lambda
		}
	}
}

// WithPPOValueCoef sets the value-loss coefficient c1 (default 0.5). Negative
// values are ignored; 0 freezes the critic.
func WithPPOValueCoef(c float64) PPOOption {
	return func(p *PPO) {
		if c >= 0 {
			p.ValueCoef = c
		}
	}
}

// WithPPOEntropyCoef sets the entropy-bonus coefficient c2 (default 0.01). Negative
// values are ignored; 0 removes the entropy term.
func WithPPOEntropyCoef(c float64) PPOOption {
	return func(p *PPO) {
		if c >= 0 {
			p.EntropyCoef = c
		}
	}
}

// WithPPORolloutSteps sets the minimum number of transitions collected per Update
// (default 256). Values ≤0 are ignored. Collection always finishes the running
// episode, so the actual batch is at least this many steps.
func WithPPORolloutSteps(n int) PPOOption {
	return func(p *PPO) {
		if n > 0 {
			p.RolloutSteps = n
		}
	}
}

// NewPPO builds a PPO agent for env with a hidden layer of the given width, Adam
// learning rate lr and discount γ, seeded deterministically. Defaults follow the
// paper/common practice (ε=0.2, K=4, λ=0.95, c1=0.5, c2=0.01, 256-step rollouts)
// and can be overridden with the With* options.
func NewPPO(env Env, hidden int, lr, gamma float64, seed uint64, opts ...PPOOption) *PPO {
	actor := policyNet(env.ObsDim(), hidden, env.NumActions(), seed)
	critic := nn.NewSequential(
		nn.NewLinear(tensor.F64, env.ObsDim(), hidden, seed+2),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, hidden, 1, seed+3),
	)
	p := &PPO{
		Actor: actor, Critic: critic,
		Gamma: gamma, Lambda: 0.95, Clip: 0.2, Epochs: 4,
		ValueCoef: 0.5, EntropyCoef: 0.01, RolloutSteps: 256,
		numActions: env.NumActions(),
		rng:        rand.New(rand.NewPCG(seed, 0x9b0)),
		aOpt:       nn.NewAdam(actor.Params(), lr),
		cOpt:       nn.NewAdam(critic.Params(), lr),
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

// Update collects one on-policy rollout, computes GAE advantages/returns from the
// critic's rollout-time values, then runs Epochs clipped policy steps and value
// regression steps. It returns the average completed-episode return in the
// rollout — the learning-curve signal that converges to the environment's optimal
// return as training proceeds.
func (p *PPO) Update(env Env) (float64, error) {
	ro, err := rlRollout(env, p.Actor, p.Critic, p.rng, p.RolloutSteps)
	if err != nil {
		return 0, err
	}
	adv, rets := GAE(ro.rewards, ro.values, ro.dones, 0, true, p.Gamma, p.Lambda)
	advT := rlNormAdv(adv)
	oldT := rlVec(ro.logpOld)

	for range p.Epochs {
		tape := autograd.NewTape()
		ctx := tape.Context()
		loss, err := ppoPolicyLoss(ctx, p.Actor, ro.states, ro.actions, oldT, advT, p.Clip, p.EntropyCoef)
		if err != nil {
			return 0, err
		}
		if err := tape.Backward(loss); err != nil {
			return 0, err
		}
		if err := p.aOpt.Step(tape.Grad); err != nil {
			return 0, err
		}
	}
	if err := rlValueFit(p.Critic, p.cOpt, ro.states, rets, p.ValueCoef, p.Epochs); err != nil {
		return 0, err
	}
	return ro.meanReturn(), nil
}

// Value returns the critic's estimate Vφ(obs) for a single observation — the
// learned state value, handy for inspecting how well the critic has fit the
// discounted return.
func (p *PPO) Value(obs []float64) (float64, error) {
	v, err := forward(backend.NewContext(), p.Critic, [][]float64{obs})
	if err != nil {
		return 0, err
	}
	return v.AtF64(0, 0), nil
}

// ActionProbs returns the current softmax policy πθ(·|obs) for a single
// observation — the post-training policy to inspect which action the agent favors.
func (p *PPO) ActionProbs(obs []float64) ([]float64, error) {
	logits, err := forward(backend.NewContext(), p.Actor, [][]float64{obs})
	if err != nil {
		return nil, err
	}
	probs, err := backend.Execute(backend.NewContext(), backend.OpSoftmax, []*tensor.Tensor{logits}, nil)
	if err != nil {
		return nil, err
	}
	out := make([]float64, p.numActions)
	for a := range p.numActions {
		out[a] = probs[0].AtF64(0, a)
	}
	return out, nil
}

// --- shared rollout / GAE / loss machinery (used by PPO and A2C) ---

// rollout holds one on-policy batch of transitions plus the completed-episode
// returns gathered while collecting it.
type rollout struct {
	states    [][]float64 // observations s_t
	actions   []int       // sampled actions a_t
	logpOld   []float64   // log πθ_old(a_t|s_t) at collection time (frozen for the ratio)
	rewards   []float64   // rewards r_t
	values    []float64   // critic values V(s_t) at collection time
	dones     []bool      // whether the episode terminated after step t
	epReturns []float64   // undiscounted returns of the episodes completed in this rollout
}

// meanReturn is the average completed-episode return (0 if none completed).
func (r *rollout) meanReturn() float64 {
	if len(r.epReturns) == 0 {
		return 0
	}
	var s float64
	for _, v := range r.epReturns {
		s += v
	}
	return s / float64(len(r.epReturns))
}

// rlRollout runs the actor greedily-sampled against env, recording transitions and
// the critic's values, until at least steps transitions are gathered; it always
// finishes the running episode so the batch ends at a terminal state (bootstrap 0).
func rlRollout(env Env, actor, critic *nn.Sequential, rng *rand.Rand, steps int) (*rollout, error) {
	k := env.NumActions()
	ro := &rollout{}
	for len(ro.states) < steps {
		obs := env.Reset()
		var epR float64
		for {
			logits, err := forward(backend.NewContext(), actor, [][]float64{obs})
			if err != nil {
				return nil, err
			}
			probs, err := backend.Execute(backend.NewContext(), backend.OpSoftmax, []*tensor.Tensor{logits}, nil)
			if err != nil {
				return nil, err
			}
			a := sampleAction(probs[0], 0, k, rng)
			v, err := forward(backend.NewContext(), critic, [][]float64{obs})
			if err != nil {
				return nil, err
			}
			ro.states = append(ro.states, obs)
			ro.actions = append(ro.actions, a)
			ro.logpOld = append(ro.logpOld, math.Log(probs[0].AtF64(0, a)))
			ro.values = append(ro.values, v.AtF64(0, 0))
			next, rew, done := env.Step(a)
			ro.rewards = append(ro.rewards, rew)
			ro.dones = append(ro.dones, done)
			epR += rew
			obs = next
			if done {
				ro.epReturns = append(ro.epReturns, epR)
				break
			}
		}
	}
	return ro, nil
}

// rlNormAdv builds the [N] advantage tensor, standardized to zero mean / unit
// variance (standard PPO practice — stabilizes the policy-gradient scale).
func rlNormAdv(adv []float64) *tensor.Tensor {
	n := len(adv)
	var m float64
	for _, a := range adv {
		m += a
	}
	m /= float64(n)
	var s float64
	for _, a := range adv {
		s += (a - m) * (a - m)
	}
	s = math.Sqrt(s/float64(n)) + 1e-8
	t := tensor.New(tensor.F64, tensor.Shape{n})
	for i, a := range adv {
		t.SetF64((a-m)/s, i)
	}
	return t
}

// rlVec builds a rank-1 tensor from a slice.
func rlVec(x []float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{len(x)})
	for i, v := range x {
		t.SetF64(v, i)
	}
	return t
}

// rlValueFit runs `epochs` critic MSE-regression steps toward the GAE returns,
// scaled by valueCoef (OpMul so a zero coefficient cleanly freezes the critic, §B60).
func rlValueFit(critic *nn.Sequential, opt nn.Optimizer, states [][]float64, returns []float64, valueCoef float64, epochs int) error {
	retT := tensor.New(tensor.F64, tensor.Shape{len(returns), 1})
	for i, r := range returns {
		retT.SetF64(r, i, 0)
	}
	for range epochs {
		tape := autograd.NewTape()
		ctx := tape.Context()
		v, err := forward(ctx, critic, states)
		if err != nil {
			return err
		}
		mse, err := nn.MSE(ctx, v, retT)
		if err != nil {
			return err
		}
		loss, err := rlScale(ctx, mse, valueCoef)
		if err != nil {
			return err
		}
		if err := tape.Backward(loss); err != nil {
			return err
		}
		if err := opt.Step(tape.Grad); err != nil {
			return err
		}
	}
	return nil
}

// rlScale multiplies a (scalar) tensor by a constant via OpMul (§B60: OpMul, not
// OpAXPY, for a possibly-zero scale), keeping the constant off the tape.
func rlScale(ctx *backend.Context, x *tensor.Tensor, c float64) (*tensor.Tensor, error) {
	k := tensor.Full(x.Dtype(), x.Shape(), c)
	out, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, k}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// rlNegEntropy returns the scalar mean_b Σ_a p_a·log p_a over the softmax policy
// rows — i.e. the NEGATIVE mean policy entropy (−H). Adding entropyCoef·(−H) to a
// minimized loss therefore MAXIMIZES entropy (the exploration bonus). Differentiable.
//
// The log-probability is the STABLE log-softmax (logits − max − log Σ exp) rather
// than log(softmax(logits)): for a near-deterministic policy a non-max softmax
// entry underflows to exactly 0, and log(0) = −∞ would make the 0·log0 entropy
// term 0·(−∞) = NaN (both the value and, via 1/p in log's VJP, the gradient). The
// stable log-softmax is always finite for finite logits, so the underflowed entry
// contributes p·logp = 0·(finite) = 0 — the correct x·log x → 0 limit (§B).
func rlNegEntropy(ctx *backend.Context, logits *tensor.Tensor) (*tensor.Tensor, error) {
	ex := func(op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, op, ins, attrs)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	sm, err := ex(backend.OpSoftmax, nil, logits)
	if err != nil {
		return nil, err
	}
	// Stable log-softmax = z − log Σ exp(z), z = logits − rowmax (all finite for
	// finite logits), avoiding the log(0) = −∞ of log(softmax).
	lastAxis := backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}
	mx, err := ex(backend.OpMax, lastAxis, logits)
	if err != nil {
		return nil, err
	}
	z, err := ex(backend.OpSub, nil, logits, mx)
	if err != nil {
		return nil, err
	}
	e, err := ex(backend.OpExp, nil, z)
	if err != nil {
		return nil, err
	}
	sums, err := ex(backend.OpSum, lastAxis, e)
	if err != nil {
		return nil, err
	}
	lse, err := ex(backend.OpLog, nil, sums)
	if err != nil {
		return nil, err
	}
	lg, err := ex(backend.OpSub, nil, z, lse)
	if err != nil {
		return nil, err
	}
	plogp, err := ex(backend.OpMul, nil, sm, lg)
	if err != nil {
		return nil, err
	}
	rows, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{1}}, plogp)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMean, nil, rows)
}

// ppoPolicyLoss builds the differentiable PPO policy objective for one batch:
// the clipped surrogate (nn.PPOClipLoss over the per-state action log-probs) minus
// entropyCoef·H. Shared by PPO.Update and the gradcheck/collapse tests so the exact
// math is one implementation.
func ppoPolicyLoss(ctx *backend.Context, actor *nn.Sequential, states [][]float64, actions []int, oldT, advT *tensor.Tensor, clip, entropyCoef float64) (*tensor.Tensor, error) {
	logits, err := forward(ctx, actor, states)
	if err != nil {
		return nil, err
	}
	lp, err := nn.TokenLogProbs(ctx, logits, actions)
	if err != nil {
		return nil, err
	}
	loss, err := nn.PPOClipLoss(ctx, lp, oldT, advT, clip)
	if err != nil {
		return nil, err
	}
	if entropyCoef != 0 {
		negH, err := rlNegEntropy(ctx, logits)
		if err != nil {
			return nil, err
		}
		term, err := rlScale(ctx, negH, entropyCoef) // entropyCoef·(−H) = −c2·H
		if err != nil {
			return nil, err
		}
		out, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{loss, term}, nil)
		if err != nil {
			return nil, err
		}
		loss = out[0]
	}
	return loss, nil
}
