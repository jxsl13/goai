package rl

import (
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// --- SAC (Soft Actor-Critic, Haarnoja et al. 2018, arXiv:1801.01290) ---

// logStdMin, logStdMax bound the actor's predicted log-standard-deviation, keeping
// the policy variance in a sane range (the SpinningUp values). logStdMin stops the
// std collapsing to 0 (infinite log-prob); logStdMax stops it exploding.
const (
	logStdMin = -20.0
	logStdMax = 2.0
)

// SAC is the Soft Actor-Critic agent for continuous control: an off-policy
// actor-critic that maximises the ENTROPY-REGULARISED return E[Σ r + α·H(π(·|s))],
// trading reward for exploration. Three ingredients distinguish it from [DDPG]:
//
//   - a STOCHASTIC squashed-Gaussian actor: the network emits a per-dimension mean
//     and log-std, an action is reparameterised u = μ + σ⊙ε and squashed a = tanh(u)
//     (then rescaled into [ActionBounds]); its log-probability carries the tanh
//     change-of-variables correction −Σ log(1 − tanh²(u)) (see [SAC.sampleActions]).
//   - TWIN critics Q1,Q2 with CLIPPED double-Q: every bootstrap uses min(Q1',Q2'),
//     which counteracts the value-overestimation bias of a single critic.
//   - an ENTROPY TEMPERATURE α weighting the policy entropy, either fixed or
//     automatically tuned toward a target entropy −ActionDim (WithSACAutoAlpha).
//
// The soft Bellman target is y = r + γ·(min(Q1',Q2') − α·log π(a'|s')) with a'
// sampled from the CURRENT policy; the actor minimises mean(α·log π(a|s) − min(Q1,Q2)).
// Targets are Polyak-averaged like DDPG (see [softUpdate]). Networks are composed
// from nn.*; the agent is deterministic given its seed. Construct with [NewSAC].
type SAC struct {
	Actor            *nn.Sequential // squashed-Gaussian actor: obs → hidden → hidden → 2·ActionDim (mean ‖ log-std)
	Critic1, Critic2 *nn.Sequential // twin action-value critics Q1,Q2: [obs;action] → hidden → hidden → 1
	Target1, Target2 *nn.Sequential // slow Polyak targets Q1',Q2' of the twin critics
	Gamma            float64        // discount factor γ in the soft Bellman target
	Tau              float64        // Polyak soft-update rate τ for the critic targets
	Alpha            float64        // entropy temperature α (fixed, or the current value under auto-tuning)
	AutoAlpha        bool           // if true, α is tuned online toward the target entropy −ActionDim
	BatchSize        int            // replay minibatch size per learning step

	obsDim, actDim int
	low, high      []float64
	scale, center  *tensor.Tensor // [1,actDim] affine mapping tanh-space → action box
	logScaleSum    float64        // Σ log(scale_i): the affine Jacobian offset in the action-space log-prob
	targetEntropy  float64        // −ActionDim, the auto-α entropy target
	logAlpha       *tensor.Tensor // [1] log α (auto-tuning only)
	buf            *replayBuffer
	rng            *rand.Rand
	aOpt, cOpt     nn.Optimizer // actor and (shared) twin-critic optimizers
	alphaOpt       nn.Optimizer // α optimizer (auto-tuning only)
}

// sacConfig collects the constructor knobs before the networks are built.
type sacConfig struct {
	hidden            int
	lr                float64
	gamma, tau, alpha float64
	autoAlpha         bool
	bufferSize, batch int
	seed              uint64
}

// SACOption configures a [SAC] agent at construction (functional-options pattern,
// §C12). All knobs carry research-grounded defaults (see [NewSAC]); each option
// validates its argument and ignores out-of-range values.
type SACOption func(*sacConfig)

// WithSACHidden sets the width of both hidden layers in the actor and critics
// (default 256, the SAC paper / SpinningUp size). Values ≤0 are ignored.
func WithSACHidden(h int) SACOption {
	return func(c *sacConfig) {
		if h > 0 {
			c.hidden = h
		}
	}
}

// WithSACLR sets the shared Adam learning rate for the actor, the twin critics and
// the entropy temperature (default 3e-4, the SAC paper value). Values ≤0 are
// ignored; SAC uses one rate for all components, unlike DDPG's split rates.
func WithSACLR(lr float64) SACOption {
	return func(c *sacConfig) {
		if lr > 0 {
			c.lr = lr
		}
	}
}

// WithSACGamma sets the discount factor γ∈[0,1] (default 0.99). Values outside
// [0,1] are ignored. γ→1 emphasises long-horizon return; γ→0 is myopic.
func WithSACGamma(g float64) SACOption {
	return func(c *sacConfig) {
		if g >= 0 && g <= 1 {
			c.gamma = g
		}
	}
}

// WithSACTau sets the Polyak soft-update rate τ∈(0,1] for the critic targets
// (default 0.005). Values outside (0,1] are ignored. SPECIAL VALUE: τ=1 is a hard
// copy each step; τ→0 freezes the targets.
func WithSACTau(tau float64) SACOption {
	return func(c *sacConfig) {
		if tau > 0 && tau <= 1 {
			c.tau = tau
		}
	}
}

// WithSACAlpha sets the entropy temperature α≥0 (default 0.2, the SAC paper's fixed
// value). Negative values are ignored. Higher α favours exploration (more stochastic
// policy); α=0 removes the entropy bonus (greedy, DDPG-like). Ignored once
// WithSACAutoAlpha(true) is set, which learns α instead.
func WithSACAlpha(alpha float64) SACOption {
	return func(c *sacConfig) {
		if alpha >= 0 {
			c.alpha = alpha
		}
	}
}

// WithSACAutoAlpha enables (or disables) automatic entropy-temperature tuning
// (default false). When enabled, α is adjusted online so the policy's average
// entropy tracks the target −ActionDim (Haarnoja et al. 2018 §5), removing the need
// to hand-pick α; the initial α is the WithSACAlpha value.
func WithSACAutoAlpha(on bool) SACOption {
	return func(c *sacConfig) { c.autoAlpha = on }
}

// WithSACBufferSize sets the replay-buffer capacity in transitions (default
// 100000). Values ≤0 are ignored. Larger buffers decorrelate updates (the paper
// uses 1e6); smaller ones forget faster.
func WithSACBufferSize(n int) SACOption {
	return func(c *sacConfig) {
		if n > 0 {
			c.bufferSize = n
		}
	}
}

// WithSACBatchSize sets the replay minibatch size per learning step (default 256,
// the SAC paper value). Values ≤0 are ignored.
func WithSACBatchSize(n int) SACOption {
	return func(c *sacConfig) {
		if n > 0 {
			c.batch = n
		}
	}
}

// WithSACSeed sets the seed for all randomness — initialisation, replay sampling
// and the reparameterisation noise — for a reproducible run (default 0).
func WithSACSeed(seed uint64) SACOption {
	return func(c *sacConfig) { c.seed = seed }
}

// NewSAC builds a Soft Actor-Critic agent for a continuous-action env. Defaults
// follow the SAC paper and OpenAI SpinningUp (hidden 256, LR 3e-4, γ=0.99, τ=0.005,
// fixed α=0.2, replay 100000, batch 256, auto-α off) and are overridable with the
// With* options. The twin critics are initialised with distinct seeds (so Q1≠Q2)
// and their targets are exact copies at construction.
func NewSAC(env ContinuousEnv, opts ...SACOption) *SAC {
	cfg := sacConfig{
		hidden: 256, lr: 3e-4, gamma: 0.99, tau: 0.005, alpha: 0.2,
		bufferSize: 100000, batch: 256,
	}
	for _, o := range opts {
		o(&cfg)
	}
	obsDim, actDim := env.ObsDim(), env.ActionDim()
	low, high := env.ActionBounds()
	scale, center := boundsScaleCenter(low, high)
	var logScaleSum float64
	//perfscan:ignore PS3010 NewSAC constructor log-sum over actionDim, one-time
	for i := range low {
		logScaleSum += math.Log((high[i] - low[i]) / 2)
	}

	actor := contMLP(obsDim, cfg.hidden, 2*actDim, cfg.seed)
	c1 := contMLP(obsDim+actDim, cfg.hidden, 1, cfg.seed+100)
	c2 := contMLP(obsDim+actDim, cfg.hidden, 1, cfg.seed+200)
	t1 := contMLP(obsDim+actDim, cfg.hidden, 1, cfg.seed+100)
	t2 := contMLP(obsDim+actDim, cfg.hidden, 1, cfg.seed+200)

	s := &SAC{
		Actor: actor, Critic1: c1, Critic2: c2, Target1: t1, Target2: t2,
		Gamma: cfg.gamma, Tau: cfg.tau, Alpha: cfg.alpha, AutoAlpha: cfg.autoAlpha,
		BatchSize: cfg.batch,
		obsDim:    obsDim, actDim: actDim, low: low, high: high,
		scale: scale, center: center, logScaleSum: logScaleSum,
		targetEntropy: -float64(actDim),
		buf:           newReplayBuffer(cfg.bufferSize, rand.New(rand.NewPCG(cfg.seed, 0x5acb))),
		rng:           rand.New(rand.NewPCG(cfg.seed, 0x5ac)),
		aOpt:          nn.NewAdam(actor.Params(), cfg.lr),
		cOpt:          nn.NewAdam(append(append([]*tensor.Tensor{}, c1.Params()...), c2.Params()...), cfg.lr),
	}
	if cfg.autoAlpha {
		s.logAlpha = tensor.New(tensor.F64, tensor.Shape{1})
		s.logAlpha.SetF64(math.Log(math.Max(cfg.alpha, 1e-8)), 0)
		s.alphaOpt = nn.NewAdam([]*tensor.Tensor{s.logAlpha}, cfg.lr)
	}
	softUpdate(c1, t1, 1)
	softUpdate(c2, t2, 1)
	return s
}

// contConst returns a [1,1] constant tensor holding c, which broadcasts over any
// [batch,dim] tensor in an elementwise op. Constants stay off the tape.
func contConst(c float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{1, 1})
	t.SetF64(c, 0, 0)
	return t
}

// sampleNoise draws an n×ActionDim tensor of independent standard-normal values
// from the agent's RNG — the reparameterisation noise ε for a batch.
func (s *SAC) sampleNoise(n int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{n, s.actDim})
	for i := 0; i < n; i++ {
		//perfscan:ignore PS1005 noise fill dominated by NormFloat64 RNG per element
		for j := 0; j < s.actDim; j++ {
			t.SetF64(s.rng.NormFloat64(), i, j)
		}
	}
	return t
}

// sacLogProb computes the squashed-Gaussian log-probability log π(a|s) for a batch,
// given the pre-squash Gaussian parameters (mean, logStd) and the pre-squash sample
// u = mean + σ⊙ε. It sums the diagonal-Gaussian log-density of u, subtracts the tanh
// change-of-variables term Σ log(1 − tanh²(u)) — in the numerically stable form
// 2·(log 2 − u − softplus(−2u)), finite even as tanh(u)→±1 — and subtracts the
// affine-rescale Jacobian Σ log(scale). Returns a [batch,1] tensor, differentiable
// through mean, logStd and u.
func (s *SAC) sacLogProb(ctx *backend.Context, mean, logStd, u *tensor.Tensor) (*tensor.Tensor, error) {
	ex := func(op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
		return contExec(ctx, op, attrs, ins...)
	}
	// Diagonal-Gaussian log-density of u: −0.5·z² − logStd − 0.5·log(2π), z=(u−μ)/σ.
	negLogStd, err := ex(backend.OpNeg, nil, logStd)
	if err != nil {
		return nil, err
	}
	invStd, err := ex(backend.OpExp, nil, negLogStd) // 1/σ = exp(−logStd)
	if err != nil {
		return nil, err
	}
	diff, err := ex(backend.OpSub, nil, u, mean)
	if err != nil {
		return nil, err
	}
	z, err := ex(backend.OpMul, nil, diff, invStd)
	if err != nil {
		return nil, err
	}
	z2, err := ex(backend.OpMul, nil, z, z)
	if err != nil {
		return nil, err
	}
	term1, err := ex(backend.OpMul, nil, z2, contConst(-0.5)) // −0.5 z²
	if err != nil {
		return nil, err
	}
	sumAB, err := ex(backend.OpAdd, nil, term1, negLogStd) // −0.5 z² − logStd
	if err != nil {
		return nil, err
	}
	gauss, err := ex(backend.OpAdd, nil, sumAB, contConst(-0.5*math.Log(2*math.Pi)))
	if err != nil {
		return nil, err
	}
	// tanh Jacobian: log(1 − tanh²u) = 2·(log2 − u − softplus(−2u)).
	twoU, err := ex(backend.OpMul, nil, u, contConst(2))
	if err != nil {
		return nil, err
	}
	negTwoU, err := ex(backend.OpNeg, nil, twoU)
	if err != nil {
		return nil, err
	}
	sp, err := ex(backend.OpSoftplus, nil, negTwoU)
	if err != nil {
		return nil, err
	}
	log2minusU, err := ex(backend.OpSub, nil, contConst(math.Ln2), u) // log2 − u
	if err != nil {
		return nil, err
	}
	inner, err := ex(backend.OpSub, nil, log2minusU, sp) // log2 − u − softplus(−2u)
	if err != nil {
		return nil, err
	}
	logDetJac, err := ex(backend.OpMul, nil, inner, contConst(2)) // log(1 − tanh²u)
	if err != nil {
		return nil, err
	}
	perDim, err := ex(backend.OpSub, nil, gauss, logDetJac) // per-dimension log-prob
	if err != nil {
		return nil, err
	}
	summed, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, perDim)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSub, nil, summed, contConst(s.logScaleSum)) // − Σ log(scale)
}

// sampleActions runs the actor on states, reparameterises a stochastic action with
// the standard-normal noise eps ([batch,ActionDim]), squashes it into the action
// box (a = center + scale⊙tanh(u)), and returns (action, logProb) — both
// differentiable through ctx (the reparameterisation trick lets the actor gradient
// flow through the sampled action).
func (s *SAC) sampleActions(ctx *backend.Context, states, eps *tensor.Tensor) (action, logp *tensor.Tensor, err error) {
	raw, err := s.Actor.Forward(ctx, states) // [batch, 2·ActionDim]
	if err != nil {
		return nil, nil, err
	}
	mean, err := contExec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: s.actDim}, raw)
	if err != nil {
		return nil, nil, err
	}
	logStdRaw, err := contExec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: s.actDim, End: 2 * s.actDim}, raw)
	if err != nil {
		return nil, nil, err
	}
	logStd, err := contExec(ctx, backend.OpClip, backend.ClipAttrs{Lo: logStdMin, Hi: logStdMax}, logStdRaw)
	if err != nil {
		return nil, nil, err
	}
	std, err := contExec(ctx, backend.OpExp, nil, logStd)
	if err != nil {
		return nil, nil, err
	}
	noise, err := contExec(ctx, backend.OpMul, nil, std, eps)
	if err != nil {
		return nil, nil, err
	}
	u, err := contExec(ctx, backend.OpAdd, nil, mean, noise)
	if err != nil {
		return nil, nil, err
	}
	logp, err = s.sacLogProb(ctx, mean, logStd, u)
	if err != nil {
		return nil, nil, err
	}
	tu, err := contExec(ctx, backend.OpTanh, nil, u)
	if err != nil {
		return nil, nil, err
	}
	scaled, err := contExec(ctx, backend.OpMul, nil, tu, s.scale)
	if err != nil {
		return nil, nil, err
	}
	action, err = contExec(ctx, backend.OpAdd, nil, scaled, s.center)
	if err != nil {
		return nil, nil, err
	}
	return action, logp, nil
}

// sacCriticTargets computes the frozen soft Bellman targets for a batch (no tape):
// y = r + γ·(1−done)·(min(Q1',Q2') − α·log π(a'|s')), with a' sampled from the
// CURRENT policy and the twin TARGET critics providing the clipped value.
func (s *SAC) sacCriticTargets(next *tensor.Tensor, rewards []float64, dones []bool) (*tensor.Tensor, error) {
	ctx := backend.NewContext()
	eps := s.sampleNoise(len(rewards))
	a2, logp2, err := s.sampleActions(ctx, next, eps)
	if err != nil {
		return nil, err
	}
	q1, err := criticForward(ctx, s.Target1, next, a2)
	if err != nil {
		return nil, err
	}
	q2, err := criticForward(ctx, s.Target2, next, a2)
	if err != nil {
		return nil, err
	}
	y := tensor.New(tensor.F64, tensor.Shape{len(rewards), 1})
	for i := range rewards {
		//perfscan:ignore PS3082 batch-256 Bellman combine, dominated by twin-critic MLP forwards
		minQ := math.Min(q1.AtF64(i, 0), q2.AtF64(i, 0)) // clipped double-Q
		soft := minQ - s.Alpha*logp2.AtF64(i, 0)
		v := rewards[i]
		if !dones[i] {
			v += s.Gamma * soft
		}
		y.SetF64(v, i, 0)
	}
	return y, nil
}

// sacCriticLoss is the twin-critic MSE Bellman loss on a fixed batch:
// MSE(Q1(s,a), y) + MSE(Q2(s,a), y). Shared by the learning step and the
// finite-difference gradcheck.
func (s *SAC) sacCriticLoss(ctx *backend.Context, states, actions, y *tensor.Tensor) (*tensor.Tensor, error) {
	q1, err := criticForward(ctx, s.Critic1, states, actions)
	if err != nil {
		return nil, err
	}
	q2, err := criticForward(ctx, s.Critic2, states, actions)
	if err != nil {
		return nil, err
	}
	l1, err := nn.MSE(ctx, q1, y)
	if err != nil {
		return nil, err
	}
	l2, err := nn.MSE(ctx, q2, y)
	if err != nil {
		return nil, err
	}
	return contExec(ctx, backend.OpAdd, nil, l1, l2)
}

// sacActorLoss is the entropy-regularised policy objective on a fixed batch:
// mean(α·log π(a|s) − min(Q1(s,a),Q2(s,a))) with a reparameterised from noise eps.
// It also returns the batch-mean log-probability (a float, for the α update).
func (s *SAC) sacActorLoss(ctx *backend.Context, states, eps *tensor.Tensor) (*tensor.Tensor, float64, error) {
	a, logp, err := s.sampleActions(ctx, states, eps)
	if err != nil {
		return nil, 0, err
	}
	q1, err := criticForward(ctx, s.Critic1, states, a)
	if err != nil {
		return nil, 0, err
	}
	q2, err := criticForward(ctx, s.Critic2, states, a)
	if err != nil {
		return nil, 0, err
	}
	minQ, err := contExec(ctx, backend.OpMinimum, nil, q1, q2) // clipped double-Q
	if err != nil {
		return nil, 0, err
	}
	alphaLogp, err := contExec(ctx, backend.OpMul, nil, logp, contConst(s.Alpha))
	if err != nil {
		return nil, 0, err
	}
	diff, err := contExec(ctx, backend.OpSub, nil, alphaLogp, minQ)
	if err != nil {
		return nil, 0, err
	}
	loss, err := contExec(ctx, backend.OpMean, nil, diff)
	if err != nil {
		return nil, 0, err
	}
	var sum float64
	n := logp.Shape()[0]
	//perfscan:ignore PS1001 batch logp mean reduction, dominated by MLP forwards
	for i := 0; i < n; i++ {
		sum += logp.AtF64(i, 0)
	}
	return loss, sum / float64(n), nil
}

// updateAlpha takes one gradient step on the entropy temperature (auto-tuning):
// minimise −log α·(meanLogp + targetEntropy), driving the policy's average entropy
// −meanLogp toward −targetEntropy = ActionDim. Updates Alpha = exp(logAlpha).
func (s *SAC) updateAlpha(meanLogp float64) error {
	c := meanLogp + s.targetEntropy // constant w.r.t. logAlpha
	tape := autograd.NewTape()
	ctx := tape.Context()
	cT := tensor.New(tensor.F64, tensor.Shape{1})
	cT.SetF64(c, 0)
	prod, err := contExec(ctx, backend.OpMul, nil, s.logAlpha, cT)
	if err != nil {
		return err
	}
	loss, err := contExec(ctx, backend.OpNeg, nil, prod)
	if err != nil {
		return err
	}
	if err := tape.Backward(loss); err != nil {
		return err
	}
	if err := s.alphaOpt.Step(tape.Grad); err != nil {
		return err
	}
	s.Alpha = math.Exp(s.logAlpha.AtF64(0))
	return nil
}

// learn samples one replay minibatch and applies a twin-critic step, an actor step,
// an optional α step and the Polyak target updates. It no-ops until the buffer holds
// at least BatchSize entries.
func (s *SAC) learn() error {
	if s.buf.len() < s.BatchSize {
		return nil
	}
	batch := s.buf.sample(s.BatchSize)
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

	y, err := s.sacCriticTargets(nextsT, rewards, dones)
	if err != nil {
		return err
	}

	// Twin-critic step.
	cTape := autograd.NewTape()
	cLoss, err := s.sacCriticLoss(cTape.Context(), statesT, actionsT, y)
	if err != nil {
		return err
	}
	if err := cTape.Backward(cLoss); err != nil {
		return err
	}
	if err := s.cOpt.Step(cTape.Grad); err != nil {
		return err
	}

	// Actor step (fresh reparameterisation noise).
	eps := s.sampleNoise(len(batch))
	aTape := autograd.NewTape()
	aLoss, meanLogp, err := s.sacActorLoss(aTape.Context(), statesT, eps)
	if err != nil {
		return err
	}
	if err := aTape.Backward(aLoss); err != nil {
		return err
	}
	if err := s.aOpt.Step(aTape.Grad); err != nil {
		return err
	}

	// Entropy-temperature step.
	if s.AutoAlpha {
		if err := s.updateAlpha(meanLogp); err != nil {
			return err
		}
	}

	// Polyak soft update of both critic targets.
	softUpdate(s.Critic1, s.Target1, s.Tau)
	softUpdate(s.Critic2, s.Target2, s.Tau)
	return nil
}

// Act returns the DETERMINISTIC evaluation action: the squashed policy MEAN
// center + scale⊙tanh(μθ(obs)) with no sampling noise — the greedy policy to use at
// test time. The returned slice has length ActionDim and lies within [ActionBounds].
func (s *SAC) Act(obs []float64) ([]float64, error) {
	ctx := backend.NewContext()
	raw, err := s.Actor.Forward(ctx, contMat([][]float64{obs}))
	if err != nil {
		return nil, err
	}
	mean, err := contExec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: s.actDim}, raw)
	if err != nil {
		return nil, err
	}
	tu, err := contExec(ctx, backend.OpTanh, nil, mean)
	if err != nil {
		return nil, err
	}
	scaled, err := contExec(ctx, backend.OpMul, nil, tu, s.scale)
	if err != nil {
		return nil, err
	}
	a, err := contExec(ctx, backend.OpAdd, nil, scaled, s.center)
	if err != nil {
		return nil, err
	}
	out := make([]float64, s.actDim)
	for i := range out {
		out[i] = a.AtF64(0, i)
	}
	return clampToBounds(out, s.low, s.high), nil
}

// rolloutAction samples a STOCHASTIC action from the current policy (used during
// environment interaction / exploration), clamped into the action box.
func (s *SAC) rolloutAction(obs []float64) ([]float64, error) {
	ctx := backend.NewContext()
	a, _, err := s.sampleActions(ctx, contMat([][]float64{obs}), s.sampleNoise(1))
	if err != nil {
		return nil, err
	}
	out := make([]float64, s.actDim)
	for i := range out {
		out[i] = a.AtF64(0, i)
	}
	return clampToBounds(out, s.low, s.high), nil
}

// Episode runs one full episode on env: at every step it samples a stochastic
// action from the current policy, stores the transition and takes one learning
// step. It returns the undiscounted episode return — the learning-curve signal.
func (s *SAC) Episode(env ContinuousEnv) (float64, error) {
	obs := env.Reset()
	var total float64
	for {
		a, err := s.rolloutAction(obs)
		if err != nil {
			return 0, err
		}
		next, rew, done := env.Step(a)
		total += rew
		s.buf.add(obs, a, rew, next, done)
		if err := s.learn(); err != nil {
			return 0, err
		}
		obs = next
		if done {
			break
		}
	}
	return total, nil
}
