package rl

import (
	"math/rand/v2"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// --- DDPG (Deep Deterministic Policy Gradient, Lillicrap et al. 2016, arXiv:1509.02971) ---

// DDPG is the deterministic off-policy actor-critic for continuous control. A
// deterministic actor μθ(s) maps a state to an action vector (its raw network
// output is tanh-squashed into the environment's [ActionBounds] box), and a critic
// Qφ(s,a) estimates the action-value. Both are trained off-policy from a FIFO
// [replayBuffer] of past transitions, using slowly-tracking TARGET copies μ',Q':
//
//	critic:  minimise MSE(Qφ(s,a), r + γ·Q'(s', μ'(s')))     (the Bellman target)
//	actor:   maximise mean Qφ(s, μθ(s))                       (deterministic policy gradient)
//
// After each learning step the targets are Polyak-averaged toward the online nets
// (target ← τ·online + (1−τ)·target, see [softUpdate]), which stabilises the moving
// bootstrap target. Exploration adds zero-mean Gaussian noise to the actor's action
// during environment interaction (the paper also offers Ornstein-Uhlenbeck noise;
// uncorrelated Gaussian is the simpler, equally-common choice — SpinningUp's default).
// The networks are composed from nn.* exactly like [PPO]; the agent is deterministic
// given its seed.
//
// Configure with the With* options; construct with [NewDDPG]. For the twin-critic,
// entropy-regularised stochastic cousin see [SAC].
type DDPG struct {
	Actor, ActorTarget   *nn.Sequential // deterministic actor μθ(s) and its slow target μ'(s): obs → hidden → hidden → ActionDim raw (pre-squash)
	Critic, CriticTarget *nn.Sequential // action-value critic Qφ(s,a) and its slow target Q'(s,a): [obs;action] → hidden → hidden → 1
	Gamma                float64        // discount factor γ used in the Bellman target
	Tau                  float64        // Polyak soft-update rate τ for the target networks
	Noise                float64        // exploration Gaussian σ added to actions during rollout (in action units)
	BatchSize            int            // replay minibatch size per learning step

	obsDim, actDim int
	low, high      []float64
	scale, center  *tensor.Tensor // [1,actDim] affine mapping tanh-space → action box
	buf            *replayBuffer
	rng            *rand.Rand
	aOpt, cOpt     nn.Optimizer
}

// ddpgConfig collects the constructor knobs before the networks are built (their
// widths/seeds are themselves options, so resolution must precede construction).
type ddpgConfig struct {
	hidden            int
	actorLR, criticLR float64
	gamma, tau        float64
	bufferSize, batch int
	noise             float64
	seed              uint64
}

// DDPGOption configures a [DDPG] agent at construction (functional-options pattern,
// §C12). All knobs have research-grounded defaults (see [NewDDPG]); each option
// validates its argument and ignores out-of-range values.
type DDPGOption func(*ddpgConfig)

// WithDDPGHidden sets the width of both hidden layers in the actor and critic MLPs
// (default 256, the SpinningUp reference size). Values ≤0 are ignored. Wider nets
// fit more complex value/policy surfaces at higher compute cost; too narrow
// underfits.
func WithDDPGHidden(h int) DDPGOption {
	return func(c *ddpgConfig) {
		if h > 0 {
			c.hidden = h
		}
	}
}

// WithDDPGActorLR sets the Adam learning rate for the actor (default 1e-4, the DDPG
// paper value — deliberately below the critic's so the policy trails the improving
// value estimate). Values ≤0 are ignored; too large destabilises the policy.
func WithDDPGActorLR(lr float64) DDPGOption {
	return func(c *ddpgConfig) {
		if lr > 0 {
			c.actorLR = lr
		}
	}
}

// WithDDPGCriticLR sets the Adam learning rate for the critic (default 1e-3, the
// DDPG paper value). Values ≤0 are ignored; too large makes the bootstrap target
// diverge.
func WithDDPGCriticLR(lr float64) DDPGOption {
	return func(c *ddpgConfig) {
		if lr > 0 {
			c.criticLR = lr
		}
	}
}

// WithDDPGGamma sets the discount factor γ∈[0,1] (default 0.99, standard deep-RL).
// Values outside [0,1] are ignored. γ→1 weights long-horizon return (harder to
// estimate); γ→0 becomes myopic (only immediate reward).
func WithDDPGGamma(g float64) DDPGOption {
	return func(c *ddpgConfig) {
		if g >= 0 && g <= 1 {
			c.gamma = g
		}
	}
}

// WithDDPGTau sets the Polyak soft-update rate τ∈(0,1] for the target networks
// (default 0.005, the SpinningUp/paper value). Values outside (0,1] are ignored.
// SPECIAL VALUE: τ=1 makes the target an exact hard copy every step (no averaging);
// τ→0 freezes the targets (very slow tracking).
func WithDDPGTau(tau float64) DDPGOption {
	return func(c *ddpgConfig) {
		if tau > 0 && tau <= 1 {
			c.tau = tau
		}
	}
}

// WithDDPGBufferSize sets the replay-buffer capacity in transitions (default
// 100000). Values ≤0 are ignored. Larger buffers decorrelate updates and retain
// more diverse experience (the paper uses 1e6 for high-dim control); smaller ones
// forget quickly and can overfit recent data.
func WithDDPGBufferSize(n int) DDPGOption {
	return func(c *ddpgConfig) {
		if n > 0 {
			c.bufferSize = n
		}
	}
}

// WithDDPGBatchSize sets the replay minibatch size per learning step (default 256,
// the SpinningUp value). Values ≤0 are ignored. Larger batches lower gradient
// variance at higher per-step cost.
func WithDDPGBatchSize(n int) DDPGOption {
	return func(c *ddpgConfig) {
		if n > 0 {
			c.batch = n
		}
	}
}

// WithDDPGNoise sets the exploration Gaussian standard deviation σ added to actions
// during environment interaction (default 0.1, the SpinningUp act_noise; in action
// units, so relative to the ±ForceMax box). Negative values are ignored. SPECIAL
// VALUE: σ=0 disables exploration noise (pure deterministic actor — usually too
// greedy to learn).
func WithDDPGNoise(sigma float64) DDPGOption {
	return func(c *ddpgConfig) {
		if sigma >= 0 {
			c.noise = sigma
		}
	}
}

// WithDDPGSeed sets the seed for all randomness — network initialisation, replay
// sampling and exploration noise — making a run fully reproducible (default 0).
func WithDDPGSeed(seed uint64) DDPGOption {
	return func(c *ddpgConfig) { c.seed = seed }
}

// NewDDPG builds a DDPG agent for a continuous-action env. Defaults follow the DDPG
// paper and the OpenAI SpinningUp reference implementation (hidden 256, actor LR
// 1e-4, critic LR 1e-3, γ=0.99, τ=0.005, replay 100000, batch 256, action noise
// 0.1) and are overridable with the With* options. The actor/critic and their
// targets are initialised identically (targets are exact copies at construction).
func NewDDPG(env ContinuousEnv, opts ...DDPGOption) *DDPG {
	cfg := ddpgConfig{
		hidden: 256, actorLR: 1e-4, criticLR: 1e-3,
		gamma: 0.99, tau: 0.005, bufferSize: 100000, batch: 256, noise: 0.1,
	}
	for _, o := range opts {
		o(&cfg)
	}
	obsDim, actDim := env.ObsDim(), env.ActionDim()
	low, high := env.ActionBounds()
	scale, center := boundsScaleCenter(low, high)

	actor := contMLP(obsDim, cfg.hidden, actDim, cfg.seed)
	actorTgt := contMLP(obsDim, cfg.hidden, actDim, cfg.seed)
	critic := contMLP(obsDim+actDim, cfg.hidden, 1, cfg.seed+100)
	criticTgt := contMLP(obsDim+actDim, cfg.hidden, 1, cfg.seed+100)

	d := &DDPG{
		Actor: actor, ActorTarget: actorTgt,
		Critic: critic, CriticTarget: criticTgt,
		Gamma: cfg.gamma, Tau: cfg.tau, Noise: cfg.noise, BatchSize: cfg.batch,
		obsDim: obsDim, actDim: actDim, low: low, high: high,
		scale: scale, center: center,
		buf:  newReplayBuffer(cfg.bufferSize, rand.New(rand.NewPCG(cfg.seed, 0xddb9))),
		rng:  rand.New(rand.NewPCG(cfg.seed, 0x0d6)),
		aOpt: nn.NewAdam(actor.Params(), cfg.actorLR),
		cOpt: nn.NewAdam(critic.Params(), cfg.criticLR),
	}
	// Targets start as exact copies of the online networks (hard copy, τ=1).
	softUpdate(d.Actor, d.ActorTarget, 1)
	softUpdate(d.Critic, d.CriticTarget, 1)
	return d
}

// squashActor forwards states through actor and squashes the raw output into the
// action box: a = center + scale⊙tanh(raw). Differentiable through ctx (the actor
// gradient flows through the tanh and the affine rescale).
func (d *DDPG) squashActor(ctx *backend.Context, actor *nn.Sequential, states *tensor.Tensor) (*tensor.Tensor, error) {
	raw, err := actor.Forward(ctx, states)
	if err != nil {
		return nil, err
	}
	th, err := contExec(ctx, backend.OpTanh, nil, raw)
	if err != nil {
		return nil, err
	}
	scaled, err := contExec(ctx, backend.OpMul, nil, th, d.scale)
	if err != nil {
		return nil, err
	}
	return contExec(ctx, backend.OpAdd, nil, scaled, d.center)
}

// criticForward evaluates a critic on a [state;action] batch by concatenating the
// two along the feature axis and running the critic MLP. Differentiable through ctx
// in both inputs (so the actor gradient can flow through the action argument).
func criticForward(ctx *backend.Context, critic *nn.Sequential, states, actions *tensor.Tensor) (*tensor.Tensor, error) {
	cat, err := contExec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, states, actions)
	if err != nil {
		return nil, err
	}
	return critic.Forward(ctx, cat)
}

// ddpgCriticTargets computes the frozen Bellman targets y = r + γ·Q'(s', μ'(s'))
// for a batch (no tape: target networks only). done transitions bootstrap 0.
func (d *DDPG) ddpgCriticTargets(states2 *tensor.Tensor, rewards []float64, dones []bool) (*tensor.Tensor, error) {
	ctx := backend.NewContext()
	a2, err := d.squashActor(ctx, d.ActorTarget, states2)
	if err != nil {
		return nil, err
	}
	q2, err := criticForward(ctx, d.CriticTarget, states2, a2)
	if err != nil {
		return nil, err
	}
	y := tensor.New(tensor.F64, tensor.Shape{len(rewards), 1})
	for i := range rewards {
		v := rewards[i]
		if !dones[i] {
			v += d.Gamma * q2.AtF64(i, 0)
		}
		y.SetF64(v, i, 0)
	}
	return y, nil
}

// ddpgCriticLoss is the critic's MSE Bellman loss on a fixed batch: MSE(Qφ(s,a), y).
// Shared by the learning step and the finite-difference gradcheck so the exact
// differentiated objective is one implementation.
func (d *DDPG) ddpgCriticLoss(ctx *backend.Context, states, actions, y *tensor.Tensor) (*tensor.Tensor, error) {
	q, err := criticForward(ctx, d.Critic, states, actions)
	if err != nil {
		return nil, err
	}
	return nn.MSE(ctx, q, y)
}

// ddpgActorLoss is the deterministic-policy-gradient objective −mean Qφ(s, μθ(s))
// on a fixed batch (minimising it ascends Q). Only the actor's parameters are
// stepped from this loss even though the critic participates in the graph.
func (d *DDPG) ddpgActorLoss(ctx *backend.Context, states *tensor.Tensor) (*tensor.Tensor, error) {
	a, err := d.squashActor(ctx, d.Actor, states)
	if err != nil {
		return nil, err
	}
	q, err := criticForward(ctx, d.Critic, states, a)
	if err != nil {
		return nil, err
	}
	m, err := contExec(ctx, backend.OpMean, nil, q)
	if err != nil {
		return nil, err
	}
	return contExec(ctx, backend.OpNeg, nil, m)
}

// learn samples one replay minibatch and applies a critic step, an actor step and a
// soft target update. It no-ops until the buffer holds at least BatchSize entries.
func (d *DDPG) learn() error {
	if d.buf.len() < d.BatchSize {
		return nil
	}
	batch := d.buf.sample(d.BatchSize)
	states := make([][]float64, len(batch))
	actions := make([][]float64, len(batch))
	nexts := make([][]float64, len(batch))
	rewards := make([]float64, len(batch))
	dones := make([]bool, len(batch))
	for i, tr := range batch {
		states[i], actions[i], nexts[i] = tr.s, tr.a, tr.s2
		rewards[i], dones[i] = tr.r, tr.done
	}
	statesT, actionsT, nextsT := contMat(states), contMat(actions), contMat(nexts)

	y, err := d.ddpgCriticTargets(nextsT, rewards, dones)
	if err != nil {
		return err
	}

	// Critic step.
	cTape := autograd.NewTape()
	cLoss, err := d.ddpgCriticLoss(cTape.Context(), statesT, actionsT, y)
	if err != nil {
		return err
	}
	if err := cTape.Backward(cLoss); err != nil {
		return err
	}
	if err := d.cOpt.Step(cTape.Grad); err != nil {
		return err
	}

	// Actor step.
	aTape := autograd.NewTape()
	aLoss, err := d.ddpgActorLoss(aTape.Context(), statesT)
	if err != nil {
		return err
	}
	if err := aTape.Backward(aLoss); err != nil {
		return err
	}
	if err := d.aOpt.Step(aTape.Grad); err != nil {
		return err
	}

	// Polyak soft update of both targets.
	softUpdate(d.Critic, d.CriticTarget, d.Tau)
	softUpdate(d.Actor, d.ActorTarget, d.Tau)
	return nil
}

// Act returns the deterministic greedy action μθ(obs), squashed into the action
// box — the policy to use for evaluation (no exploration noise). The returned
// slice has length ActionDim and always lies within [ActionBounds].
func (d *DDPG) Act(obs []float64) ([]float64, error) {
	a, err := d.squashActor(backend.NewContext(), d.Actor, contMat([][]float64{obs}))
	if err != nil {
		return nil, err
	}
	out := make([]float64, d.actDim)
	for i := range out {
		out[i] = a.AtF64(0, i)
	}
	return clampToBounds(out, d.low, d.high), nil
}

// exploreAction returns the greedy action plus zero-mean Gaussian exploration
// noise (σ=Noise), clamped into the action box.
func (d *DDPG) exploreAction(obs []float64) ([]float64, error) {
	a, err := d.Act(obs)
	if err != nil {
		return nil, err
	}
	if d.Noise > 0 {
		for i := range a {
			a[i] += d.Noise * d.rng.NormFloat64()
		}
		clampToBounds(a, d.low, d.high)
	}
	return a, nil
}

// Episode runs one full episode on env: at every step it acts with exploration
// noise, stores the transition in the replay buffer and takes one learning step.
// It returns the undiscounted episode return — the learning-curve signal that
// climbs toward the environment's optimum as training proceeds.
func (d *DDPG) Episode(env ContinuousEnv) (float64, error) {
	obs := env.Reset()
	var total float64
	for {
		a, err := d.exploreAction(obs)
		if err != nil {
			return 0, err
		}
		next, rew, done := env.Step(a)
		total += rew
		d.buf.add(obs, a, rew, next, done)
		if err := d.learn(); err != nil {
			return 0, err
		}
		obs = next
		if done {
			break
		}
	}
	return total, nil
}
