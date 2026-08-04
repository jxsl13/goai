// Package rl is layer L4: reinforcement-learning basics (§T26) — REINFORCE
// (Williams 1992) and DQN (Mnih et al. 2015) on toy environments, built
// entirely on the goai nn/autograd stack. Acceptance is learning behavior on a
// deterministic MDP plus exactly-verifiable pieces (discounted returns).
package rl

import (
	"math/rand/v2"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Env is a minimal episodic environment.
type Env interface {
	Reset() []float64
	Step(action int) (obs []float64, reward float64, done bool)
	NumActions() int
	ObsDim() int
}

// Chain is the classic N-state chain MDP: the agent starts in the middle and
// moves left/right. Reaching the right end pays 1.0, the left end 0.1; the
// optimal policy is "always right". One-hot observations.
type Chain struct {
	N          int // number of states in the chain env
	MaxSteps   int // episode step cap
	pos, steps int
}

// NewChain builds a chain with n states (n ≥ 3).
func NewChain(n, maxSteps int) *Chain { return &Chain{N: n, MaxSteps: maxSteps} }

// NumActions returns the number of actions (2: left, right).
func (c *Chain) NumActions() int { return 2 }

// ObsDim returns the observation dimension (one-hot over the N states).
func (c *Chain) ObsDim() int { return c.N }

func (c *Chain) obs() []float64 {
	o := make([]float64, c.N)
	o[c.pos] = 1
	return o
}

// Reset places the agent in the middle state and returns the initial observation.
func (c *Chain) Reset() []float64 {
	c.pos = c.N / 2
	c.steps = 0
	return c.obs()
}

// Step moves left (0) or right (1).
func (c *Chain) Step(action int) ([]float64, float64, bool) {
	c.steps++
	if action == 1 {
		c.pos++
	} else {
		c.pos--
	}
	switch {
	case c.pos >= c.N-1:
		return c.obs(), 1.0, true
	case c.pos <= 0:
		return c.obs(), 0.1, true
	case c.steps >= c.MaxSteps:
		return c.obs(), 0, true
	default:
		return c.obs(), 0, false
	}
}

// DiscountedReturns computes G_t = Σ_{k≥t} γ^{k−t} r_k (exactly testable).
func DiscountedReturns(rewards []float64, gamma float64) []float64 {
	g := make([]float64, len(rewards))
	var acc float64
	for t := len(rewards) - 1; t >= 0; t-- {
		acc = rewards[t] + gamma*acc
		g[t] = acc
	}
	return g
}

// GAE computes Generalized Advantage Estimation advantages and value-function
// targets (Schulman et al. 2016, arXiv:1506.02438) — the advantages the PPO
// clipped loss (nn.PPOClipLoss) consumes. For a trajectory of length T:
//
//	δ_t = r_t + γ·V(s_{t+1})·nonterminal_t − V(s_t)
//	Â_t = δ_t + γ·λ·nonterminal_t·Â_{t+1}      (backward recursion, Â reset at episode ends)
//	returns_t = Â_t + V(s_t)
//
// which equals the exponentially-weighted sum Â_t = Σ_{l≥0} (γλ)^l δ_{t+l} within
// an episode. λ=0 gives the one-step TD advantage δ_t; λ=1 gives the Monte-Carlo
// advantage (discounted return − V). Convention (matches CleanRL/SB3): rewards,
// values and dones are aligned per step; dones[t] reports whether the episode
// terminated AFTER step t. nextValue is the bootstrap V(s_T) for the state after
// the last step and nextDone whether that state is terminal. γ default ~0.99,
// λ ~0.95. Panics on length mismatch (a programmer error).
func GAE(rewards, values []float64, dones []bool, nextValue float64, nextDone bool, gamma, lambda float64) (advantages, returns []float64) {
	t := len(rewards)
	if len(values) != t || len(dones) != t {
		panic("rl: GAE rewards/values/dones length mismatch")
	}
	advantages = make([]float64, t)
	returns = make([]float64, t)
	var lastGAE float64
	for i := t - 1; i >= 0; i-- {
		var nextNonTerminal, nextV float64
		if i == t-1 {
			if !nextDone {
				nextNonTerminal = 1
			}
			nextV = nextValue
		} else {
			if !dones[i] { // dones[i]: episode ended after step i → s_{i+1} fresh
				nextNonTerminal = 1
			}
			nextV = values[i+1]
		}
		delta := rewards[i] + gamma*nextV*nextNonTerminal - values[i]
		lastGAE = delta + gamma*lambda*nextNonTerminal*lastGAE
		advantages[i] = lastGAE
		returns[i] = lastGAE + values[i]
	}
	return advantages, returns
}

// policyNet is a small MLP obs→hidden→actions shared by both agents.
func policyNet(obsDim, hidden, actions int, seed uint64) *nn.Sequential {
	return nn.NewSequential(
		nn.NewLinear(tensor.F64, obsDim, hidden, seed),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, hidden, actions, seed+1),
	)
}

func forward(ctx *backend.Context, net *nn.Sequential, states [][]float64) (*tensor.Tensor, error) {
	n, d := len(states), len(states[0])
	x := tensor.New(tensor.F64, tensor.Shape{n, d})
	// Typed contiguous fill: x is a fresh row-major [n,d] F64 tensor, so row i occupies
	// flat[i*d:(i+1)*d]; copy each state row straight into the storage instead of the
	// per-element SetF64 dispatch the generic fill paid on all n·d inputs (§base-perf).
	xf := x.Storage().F64()
	for i, s := range states {
		copy(xf[i*d:(i+1)*d], s)
	}
	return net.Forward(ctx, x)
}

// sampleAction draws from the softmax policy over logits row i.
func sampleAction(probs *tensor.Tensor, row int, k int, rng *rand.Rand) int {
	u := rng.Float64()
	var cum float64
	for a := range k {
		cum += probs.AtF64(row, a)
		if u <= cum {
			return a
		}
	}
	return k - 1
}

// --- REINFORCE ---

// Reinforce trains a softmax policy by episodic policy gradient with a mean
// baseline: ∇J = E[Σ ∇log π(a|s)·(G−b)]. The loss is composed from existing
// differentiable ops as Sum(W ⊙ log softmax(logits)) with the constant weight
// matrix W[t, aₜ] = −(Gₜ−b)/T.
type Reinforce struct {
	Net   *nn.Sequential // policy network
	Gamma float64        // discount factor γ
	rng   *rand.Rand
	opt   nn.Optimizer

	// baseline is a cross-episode EMA of the episode return G₀. A within-episode
	// mean-of-Gₜ baseline would subtract away exactly the between-episode signal
	// REINFORCE needs (§B26) — the baseline must come from other episodes.
	baseline    float64
	baselineSet bool
}

// NewReinforce builds the agent (deterministic via seed).
func NewReinforce(env Env, hidden int, lr, gamma float64, seed uint64) *Reinforce {
	net := policyNet(env.ObsDim(), hidden, env.NumActions(), seed)
	return &Reinforce{
		Net: net, Gamma: gamma,
		rng: rand.New(rand.NewPCG(seed, 0x5eed)),
		opt: nn.NewAdam(net.Params(), lr),
	}
}

// Episode runs one episode and applies one policy-gradient step. Returns the
// (undiscounted) episode reward.
func (r *Reinforce) Episode(env Env) (float64, error) {
	k := env.NumActions()
	var states [][]float64
	var actions []int
	var rewards []float64

	obs := env.Reset()
	for {
		//perfscan:ignore PS1003 single-obs forward in rollout; net forward dwarfs slice-wrap
		logits, err := forward(backend.NewContext(), r.Net, [][]float64{obs})
		if err != nil {
			return 0, err
		}
		probs, err := backend.Execute(backend.NewContext(), backend.OpSoftmax, []*tensor.Tensor{logits}, nil)
		if err != nil {
			return 0, err
		}
		a := sampleAction(probs[0], 0, k, r.rng)
		states = append(states, obs)
		actions = append(actions, a)
		next, rew, done := env.Step(a)
		rewards = append(rewards, rew)
		obs = next
		if done {
			break
		}
	}

	returns := DiscountedReturns(rewards, r.Gamma)
	if !r.baselineSet {
		r.baseline = returns[0]
		r.baselineSet = true
	}
	baseline := r.baseline
	r.baseline = 0.9*r.baseline + 0.1*returns[0] // update EMA for future episodes

	// loss = Σ W ⊙ log softmax(logits), W[t,aₜ] = −(Gₜ−b)/T (constant)
	tape := autograd.NewTape()
	ctx := tape.Context()
	logits, err := forward(ctx, r.Net, states)
	if err != nil {
		return 0, err
	}
	sm, err := backend.Execute(ctx, backend.OpSoftmax, []*tensor.Tensor{logits}, nil)
	if err != nil {
		return 0, err
	}
	lg, err := backend.Execute(ctx, backend.OpLog, sm, nil)
	if err != nil {
		return 0, err
	}
	w := tensor.New(tensor.F64, tensor.Shape{len(states), k})
	//perfscan:ignore PS1005 T-bounded sparse scatter, dwarfed by net forward+backward
	for t, a := range actions {
		w.SetF64(-(returns[t]-baseline)/float64(len(states)), t, a)
	}
	weighted, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{w, lg[0]}, nil)
	if err != nil {
		return 0, err
	}
	loss, err := backend.Execute(ctx, backend.OpSum, weighted, nil)
	if err != nil {
		return 0, err
	}
	if err := tape.Backward(loss[0]); err != nil {
		return 0, err
	}
	if err := r.opt.Step(tape.Grad); err != nil {
		return 0, err
	}

	var total float64
	for _, v := range rewards {
		total += v
	}
	return total, nil
}

// --- DQN ---

type transition struct {
	s, s2 []float64
	a     int
	r     float64
	done  bool
}

// DQN is a minimal deep Q-learner: replay buffer, target network synced every
// SyncEvery steps, ε-greedy exploration with linear decay, and the standard
// replaced-entry MSE trick (targets equal predictions except at the taken
// action, so non-taken entries contribute zero gradient).
type DQN struct {
	Net, Target *nn.Sequential // Net = online Q-network; Target = periodically-synced copy
	Gamma       float64        // discount factor γ
	Eps, EpsMin float64        // Eps = current ε-greedy rate; EpsMin = ε floor
	EpsDecay    float64        // per-step multiplicative ε decay
	SyncEvery   int            // steps between target-net syncs
	BatchSize   int            // replay minibatch size

	buf   []transition
	steps int
	rng   *rand.Rand
	opt   nn.Optimizer
}

// NewDQN builds the agent (deterministic via seed).
func NewDQN(env Env, hidden int, lr, gamma float64, seed uint64) *DQN {
	net := policyNet(env.ObsDim(), hidden, env.NumActions(), seed)
	tgt := policyNet(env.ObsDim(), hidden, env.NumActions(), seed)
	d := &DQN{
		Net: net, Target: tgt, Gamma: gamma,
		Eps: 1.0, EpsMin: 0.05, EpsDecay: 0.995,
		SyncEvery: 25, BatchSize: 32,
		rng: rand.New(rand.NewPCG(seed, 0xd9e)),
		opt: nn.NewAdam(net.Params(), lr),
	}
	d.sync()
	return d
}

// sync copies online-net parameters into the target net.
func (d *DQN) sync() {
	src, dst := d.Net.Params(), d.Target.Params()
	for i := range src {
		s, t := src[i], dst[i]
		// Typed contiguous fast path: a hard copy target←online is one memmove per
		// parameter instead of a per-element Unravel/AtF64/SetF64 walk (§base-perf). The
		// F32→F64→F32 round-trip the generic path performed is exact, so bit-identical.
		if s.Dtype() == tensor.F64 && t.Dtype() == tensor.F64 && s.IsContiguous() && t.IsContiguous() {
			copy(t.Storage().F64()[t.Offset():t.Offset()+t.Numel()], s.Storage().F64()[s.Offset():s.Offset()+s.Numel()])
			continue
		}
		if s.Dtype() == tensor.F32 && t.Dtype() == tensor.F32 && s.IsContiguous() && t.IsContiguous() {
			copy(t.Storage().F32()[t.Offset():t.Offset()+t.Numel()], s.Storage().F32()[s.Offset():s.Offset()+s.Numel()])
			continue
		}
		for j := range s.Numel() {
			idx := tensor.Unravel(j, s.Shape())
			t.SetF64(s.AtF64(idx...), idx...)
		}
	}
}

// act picks ε-greedily from Q(s).
func (d *DQN) act(obs []float64, k int) (int, error) {
	if d.rng.Float64() < d.Eps {
		return d.rng.IntN(k), nil
	}
	q, err := forward(backend.NewContext(), d.Net, [][]float64{obs})
	if err != nil {
		return 0, err
	}
	best, bv := 0, q.AtF64(0, 0)
	//perfscan:ignore PS3068 argmax over k actions, low trip-count tiny vector
	for a := 1; a < k; a++ {
		if v := q.AtF64(0, a); v > bv {
			bv, best = v, a
		}
	}
	return best, nil
}

// Episode runs one episode with learning. Returns the episode reward.
func (d *DQN) Episode(env Env) (float64, error) {
	k := env.NumActions()
	obs := env.Reset()
	var total float64
	for {
		a, err := d.act(obs, k)
		if err != nil {
			return 0, err
		}
		next, rew, done := env.Step(a)
		total += rew
		d.buf = append(d.buf, transition{s: obs, s2: next, a: a, r: rew, done: done})
		if len(d.buf) > 4096 {
			d.buf = d.buf[1:]
		}
		if err := d.learn(k); err != nil {
			return 0, err
		}
		obs = next
		d.steps++
		if d.steps%d.SyncEvery == 0 {
			d.sync()
		}
		if done {
			break
		}
	}
	if d.Eps > d.EpsMin {
		d.Eps *= d.EpsDecay
	}
	return total, nil
}

// learn samples a batch and applies one TD(0) step.
func (d *DQN) learn(k int) error {
	if len(d.buf) < d.BatchSize {
		return nil
	}
	batch := make([]transition, d.BatchSize)
	for i := range batch {
		batch[i] = d.buf[d.rng.IntN(len(d.buf))]
	}
	states := make([][]float64, len(batch))
	nexts := make([][]float64, len(batch))
	for i, tr := range batch {
		states[i], nexts[i] = tr.s, tr.s2
	}

	// TD targets from the frozen target net (no tape)
	qNext, err := forward(backend.NewContext(), d.Target, nexts)
	if err != nil {
		return err
	}
	// ONE forward over the online net, not two. The non-taken entries of the target were filled
	// from a separate untaped forward over the SAME net and the SAME states, with no optimizer step
	// between it and the taped one below — so the two produced identical values and the first was
	// pure waste, a full MLP forward over the batch on every environment step.
	//
	// The tape is opened before the target is built rather than after. A recorder only appends
	// nodes as ops execute (it does not change execution), so the values are the same either way;
	// what changes is that the target loop can now read the taped result.
	tape := autograd.NewTape()
	ctx := tape.Context()
	q, err := forward(ctx, d.Net, states)
	if err != nil {
		return err
	}
	target := tensor.New(tensor.F64, tensor.Shape{len(batch), k})
	for i, tr := range batch {
		//perfscan:ignore PS1005 argmax over k actions, low trip-count
		for a := range k {
			target.SetF64(q.AtF64(i, a), i, a)
		}
		y := tr.r
		if !tr.done {
			best := qNext.AtF64(i, 0)
			//perfscan:ignore PS1005 batch*k target fill, dwarfed by 3 net passes per learn()
			for a := 1; a < k; a++ {
				if v := qNext.AtF64(i, a); v > best {
					best = v
				}
			}
			y += d.Gamma * best
		}
		target.SetF64(y, i, tr.a)
	}

	loss, err := nn.MSE(ctx, q, target)
	if err != nil {
		return err
	}
	if err := tape.Backward(loss); err != nil {
		return err
	}
	return d.opt.Step(tape.Grad)
}

// GreedyAction returns argmax_a Q(s,a) (post-training policy inspection).
func (d *DQN) GreedyAction(obs []float64, k int) (int, error) {
	q, err := forward(backend.NewContext(), d.Net, [][]float64{obs})
	if err != nil {
		return 0, err
	}
	best, bv := 0, q.AtF64(0, 0)
	//perfscan:ignore PS3068 GreedyAction argmax, post-training inspection cold path
	for a := 1; a < k; a++ {
		if v := q.AtF64(0, a); v > bv {
			bv, best = v, a
		}
	}
	return best, nil
}

// QValues returns Q(s,·) for inspection.
func (d *DQN) QValues(obs []float64, k int) ([]float64, error) {
	q, err := forward(backend.NewContext(), d.Net, [][]float64{obs})
	if err != nil {
		return nil, err
	}
	out := make([]float64, k)
	for a := range k {
		out[a] = q.AtF64(0, a)
	}
	return out, nil
}
