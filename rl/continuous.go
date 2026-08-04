package rl

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/parallel"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Continuous-action reinforcement learning (§T26 continuous-control extension).
//
// # For AI practitioners
//
// The discrete [Env]/[DQN]/[PPO] stack above covers environments with a finite
// action set. The other half of deep RL — robotics, locomotion, continuous
// control — needs REAL-VALUED action vectors, and a different family of
// algorithms: the off-policy actor-critic methods DDPG (Lillicrap et al. 2016,
// arXiv:1509.02971) and SAC (Haarnoja et al. 2018, arXiv:1801.01290) built on a
// replay buffer and target networks. This file adds the shared machinery:
//
//   - [ContinuousEnv] — the continuous analogue of [Env]: Step takes an action
//     VECTOR []float64 instead of an int, and the env advertises its action
//     dimensionality and per-dimension bounds ([ActionBounds]) so agents can
//     tanh-squash their outputs into the valid range.
//   - [PointMass] — a small, deterministic-enough LQR-style benchmark: a body in
//     ℝ^Dim pushed by a bounded force must reach and settle at the origin. Its
//     optimal policy is a smooth feedback law, so convergence is testable in a
//     few thousand steps and a trained agent beats a random policy by a wide
//     margin — the continuous-control counterpart of [Chain].
//
// [DDPG] and [SAC] share a FIFO replay buffer, soft target-network updates
// (τ·online + (1−τ)·target), and tanh-squashed actors composed from nn.* layers,
// all implemented here so the two agent files stay focused on their loss math.
//
// # For everyone else
//
// Some control problems don't have a handful of discrete moves ("left"/"right")
// but a continuous dial: how hard to push, what angle to steer, how much torque
// to apply. This adds a tiny practice task — a puck on a frictionless table that
// must coast to the centre and stop, steered by a continuous push — and the two
// standard algorithms that learn such dials by storing past experience and
// replaying it (DDPG and SAC).
//
// Further reading: Sutton & Barto, "Reinforcement Learning: An Introduction"
// (2nd ed., MIT Press 2018), ch. 13 (policy gradients); Lillicrap et al. 2016
// (DDPG) and Haarnoja et al. 2018 (SAC) for the two agents; OpenAI Spinning Up
// (spinningup.openai.com) for reference implementations.

// ContinuousEnv is the continuous-action analogue of [Env]: an episodic
// environment whose actions are real-valued vectors rather than discrete indices.
//
// The contract mirrors [Env]. Reset starts a new episode and returns the initial
// observation; Step applies an action VECTOR and returns the next observation, a
// scalar reward and whether the episode terminated. ActionDim and ObsDim report
// the vector widths, and ActionBounds returns the inclusive per-dimension limits
// low[i] ≤ action[i] ≤ high[i] — the box an agent must keep its actions inside
// (both slices have length ActionDim). Agents typically squash a network output
// through tanh and rescale into [low,high] so every emitted action is valid.
type ContinuousEnv interface {
	Reset() []float64                                                 // start a new episode, return the initial observation
	Step(action []float64) (obs []float64, reward float64, done bool) // apply an action vector
	ActionDim() int                                                   // length of the action vector
	ObsDim() int                                                      // length of the observation vector
	ActionBounds() (low, high []float64)                              // inclusive per-dimension action box
}

// PointMass is a deterministic-enough continuous-control benchmark: a unit point
// mass in ℝ^Dim, driven by a bounded force, that must reach and settle at the
// origin. It is the continuous counterpart of [Chain] — small, seeded and fast to
// solve, so both [DDPG] and [SAC] can be shown to converge on it in a few thousand
// steps and to beat a random policy by a wide margin.
//
// Dynamics (semi-implicit Euler with linear drag, so the map is well-conditioned
// and a simple feedback law is optimal):
//
//	v ← Damping·v + a·Dt
//	x ← x + v·Dt
//
// with the action a clamped to the ±ForceMax box. The observation is the stacked
// [position; velocity] (length 2·Dim) and the reward is the negative LQR cost
//
//	r = −(‖x‖² + 0.05·‖v‖² + 0.01·‖a‖²)
//
// which is maximised (r→0) by parking the mass motionless at the origin. Episodes
// reset the position uniformly in [−Start,Start]^Dim with zero velocity and run
// for MaxSteps steps. Determinism is fixed by the constructor seed.
type PointMass struct {
	Dim      int     // spatial dimensions (action width = Dim, observation width = 2·Dim)
	MaxSteps int     // episode step cap
	Dt       float64 // integration timestep
	Damping  float64 // per-step velocity retention ∈ [0,1] (linear drag)
	ForceMax float64 // action magnitude bound (box is [−ForceMax, ForceMax]^Dim)
	Start    float64 // reset half-width: initial position uniform in [−Start, Start]^Dim

	pos, vel []float64
	steps    int
	rng      *rand.Rand
}

// NewPointMass builds a Dim-dimensional point-mass environment with sensible,
// convergence-tested defaults (Dt=0.1, Damping=0.85, ForceMax=1, Start=1,
// MaxSteps=40) and a deterministic RNG seeded by seed. Dim must be ≥ 1.
func NewPointMass(dim int, seed uint64) *PointMass {
	if dim < 1 {
		panic(fmt.Sprintf("rl: PointMass dim must be ≥ 1, got %d", dim))
	}
	return &PointMass{
		Dim: dim, MaxSteps: 40, Dt: 0.1, Damping: 0.85, ForceMax: 1, Start: 1,
		pos: make([]float64, dim), vel: make([]float64, dim),
		rng: rand.New(rand.NewPCG(seed, 0xc047)),
	}
}

// ObsDim returns the observation width (2·Dim: position then velocity).
func (p *PointMass) ObsDim() int { return 2 * p.Dim }

// ActionDim returns the action width (Dim: one force component per axis).
func (p *PointMass) ActionDim() int { return p.Dim }

// ActionBounds returns the symmetric per-axis force box [−ForceMax, ForceMax].
func (p *PointMass) ActionBounds() (low, high []float64) {
	low = make([]float64, p.Dim)
	high = make([]float64, p.Dim)
	for i := range low {
		low[i], high[i] = -p.ForceMax, p.ForceMax
	}
	return low, high
}

func (p *PointMass) obs() []float64 {
	o := make([]float64, 2*p.Dim)
	copy(o[:p.Dim], p.pos)
	copy(o[p.Dim:], p.vel)
	return o
}

// Reset draws a fresh start position uniformly in [−Start,Start]^Dim, zeroes the
// velocity and returns the initial observation.
func (p *PointMass) Reset() []float64 {
	for i := range p.pos {
		p.pos[i] = (2*p.rng.Float64() - 1) * p.Start
		p.vel[i] = 0
	}
	p.steps = 0
	return p.obs()
}

// Step clamps the action to the force box, advances the semi-implicit Euler
// dynamics one tick, and returns the next observation, the negative-LQR-cost
// reward and whether the step cap was reached.
func (p *PointMass) Step(action []float64) ([]float64, float64, bool) {
	p.steps++
	var costX, costV, costA float64
	for i := range p.pos {
		a := action[i]
		if a > p.ForceMax {
			a = p.ForceMax
		} else if a < -p.ForceMax {
			a = -p.ForceMax
		}
		p.vel[i] = p.Damping*p.vel[i] + a*p.Dt
		p.pos[i] += p.vel[i] * p.Dt
		costX += p.pos[i] * p.pos[i]
		costV += p.vel[i] * p.vel[i]
		costA += a * a
	}
	reward := -(costX + 0.05*costV + 0.01*costA)
	return p.obs(), reward, p.steps >= p.MaxSteps
}

// --- shared continuous-control machinery (replay buffer, MLPs, target updates) ---

// contTransition is one stored (s, a, r, s', done) experience tuple.
type contTransition struct {
	s, a, s2 []float64
	r        float64
	done     bool
}

// replayBuffer is a fixed-capacity FIFO (ring) buffer of transitions shared by
// [DDPG] and [SAC]: add overwrites the oldest entry once full (capacity is a hard
// cap), and sample draws a uniform minibatch WITH replacement using the buffer's
// own seeded RNG, so a fixed seed makes the sampled batches reproducible.
type replayBuffer struct {
	data []contTransition
	cap  int
	pos  int // next write slot (ring)
	full bool
	rng  *rand.Rand
}

// newReplayBuffer builds an empty ring buffer of the given capacity (≥ 1),
// deterministic via rng.
func newReplayBuffer(capacity int, rng *rand.Rand) *replayBuffer {
	if capacity < 1 {
		capacity = 1
	}
	return &replayBuffer{data: make([]contTransition, capacity), cap: capacity, rng: rng}
}

// add stores one transition, overwriting the oldest once the capacity is reached.
func (b *replayBuffer) add(s, a []float64, r float64, s2 []float64, done bool) {
	b.data[b.pos] = contTransition{s: s, a: a, s2: s2, r: r, done: done}
	b.pos++
	if b.pos >= b.cap {
		b.pos = 0
		b.full = true
	}
}

// len returns the number of stored transitions (≤ capacity).
func (b *replayBuffer) len() int {
	if b.full {
		return b.cap
	}
	return b.pos
}

// sample draws n transitions uniformly with replacement from the stored entries.
func (b *replayBuffer) sample(n int) []contTransition {
	m := b.len()
	out := make([]contTransition, n)
	for i := range out {
		out[i] = b.data[b.rng.IntN(m)]
	}
	return out
}

// contMLP builds the standard two-hidden-layer ReLU MLP inDim→hidden→hidden→outDim
// used for the continuous actors and critics. Seeded deterministically; the three
// Linear layers get distinct seeds so their weights are independent.
func contMLP(inDim, hidden, outDim int, seed uint64) *nn.Sequential {
	return nn.NewSequential(
		nn.NewLinear(tensor.F64, inDim, hidden, seed),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, hidden, hidden, seed+1),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, hidden, outDim, seed+2),
	)
}

// contMat builds a [rows][cols] rank-2 tensor from a slice of equal-length rows.
func contMat(rows [][]float64) *tensor.Tensor {
	n, d := len(rows), len(rows[0])
	// Row-major copy straight into the storage — no SetF64 dispatch per element (PS1005).
	t := tensor.New(tensor.F64, tensor.Shape{n, d})
	ts := t.Storage().F64()
	for i, r := range rows {
		copy(ts[i*d:i*d+d], r)
	}
	return t
}

// contExec runs a single-output op through ctx and returns its lone result — the
// terse form used pervasively by the actor/critic loss builders.
func contExec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// boundsScaleCenter returns the affine map from tanh-space [−1,1] into the action
// box: scale=(high−low)/2 and center=(high+low)/2, each as a [1,dim] tensor that
// broadcasts over a [batch,dim] tanh output so squashed = center + scale⊙tanh(·).
func boundsScaleCenter(low, high []float64) (scale, center *tensor.Tensor) {
	d := len(low)
	scale = tensor.New(tensor.F64, tensor.Shape{1, d})
	center = tensor.New(tensor.F64, tensor.Shape{1, d})
	for i := range low {
		scale.SetF64((high[i]-low[i])/2, 0, i)
		center.SetF64((high[i]+low[i])/2, 0, i)
	}
	return scale, center
}

// softUpdateParThreshold is the element count above which one parameter's Polyak blend is
// worth fanning out. Each element does two reads and one write of a resident array, so the
// loop is bandwidth-bound rather than compute-bound and the crossover sits where the fan-out
// overhead stops dominating. 1<<15 matches the threshold the other packages in this
// repository use for the same kind of elementwise sweep.
//
// At the shape this actually runs on — a critic-sized MLP, hidden 256 — it fans out the two
// 65536-element weight matrices and leaves the three bias vectors of 256 or fewer elements
// serial, which is the intended split: a 256-element fan-out is pure overhead.
//
//perfscan:ignore PS6023 softUpdate already typed fastpath; finding on generic fallback
const softUpdateParThreshold = 1 << 15

// softUpdateRun applies body over [0,n) in parallel above the threshold and serially below
// it. Splitting is legal here because the blend carries NO accumulation: every output index
// is written exactly once, from its own source element and its own previous value, so chunk
// order cannot affect a single bit. That is the property to check before reusing this shape —
// a loop that summed into a shared float would reassociate and change results.
//
// Self-aliasing is safe too. Even if the source and destination were the same array, chunk j
// reads and writes only index j, so no chunk can observe another's write.
func softUpdateRun(n int, body func(lo, hi int)) {
	if n < softUpdateParThreshold || parallel.Workers() <= 1 {
		body(0, n)
		return
	}
	parallel.Rows(n, body)
}

// softUpdate performs the Polyak soft update of a target network toward an online
// one: for every parameter, target ← τ·online + (1−τ)·target. τ=1 is an exact
// hard copy (target ← online); τ=0 leaves the target unchanged. The two networks
// must share parameter shapes (they are built identically).
func softUpdate(online, target *nn.Sequential, tau float64) {
	src, dst := online.Params(), target.Params()
	for i := range src {
		s, d := src[i], dst[i]
		// Typed contiguous fast path (§base-perf: dtype-switch ONCE, no per-element
		// tensor.Unravel alloc / AtF64 / SetF64 dispatch — softUpdate runs every training
		// step over every parameter). Params are contiguous, so flat index = storage index,
		// and the arithmetic stays in float64 in the same order the generic path uses, so
		// the result is bit-identical (F32 reads widen and the store rounds exactly as
		// AtF64/SetF64 did). Mixed/non-contiguous dtypes fall through to the generic walk.
		if s.Dtype() == tensor.F64 && d.Dtype() == tensor.F64 && s.IsContiguous() && d.IsContiguous() {
			so := s.Storage().F64()[s.Offset() : s.Offset()+s.Numel()]
			to := d.Storage().F64()[d.Offset() : d.Offset()+d.Numel()]
			// omt is hoisted and the chunk is re-sliced for one reason each, and the two are
			// connected: the loop carried a bounds check on so[j] AND on to[j], and those two
			// panic edges split the body into separate basic blocks — Go's SSA will not hoist
			// across a block that can panic, so (1-tau) was rematerialized every iteration
			// (FMOVD $(1.0) then FSUBD, visible in -gcflags=-S). Discharging the checks is what
			// makes the hoist possible; ranging over ds bounds j, and cutting ss to len(ds)
			// relates the second operand to it. 1-tau computed once is the identical double,
			// and the expression shape is otherwise untouched, so nothing is reassociated.
			omt := 1 - tau
			softUpdateRun(len(to), func(lo, hi int) {
				ds, ss := to[lo:hi], so[lo:hi]
				ss = ss[:len(ds)]
				for j := range ds {
					//perfscan:ignore PS3084 rl continuous-control helper, small dims, low-leverage
					ds[j] = tau*ss[j] + omt*ds[j]
				}
			})
			continue
		}
		if s.Dtype() == tensor.F32 && d.Dtype() == tensor.F32 && s.IsContiguous() && d.IsContiguous() {
			so := s.Storage().F32()[s.Offset() : s.Offset()+s.Numel()]
			to := d.Storage().F32()[d.Offset() : d.Offset()+d.Numel()]
			// Same two defects as the F64 arm above, same remedy; see the comment there.
			omt := 1 - tau
			softUpdateRun(len(to), func(lo, hi int) {
				ds, ss := to[lo:hi], so[lo:hi]
				ss = ss[:len(ds)]
				for j := range ds {
					ds[j] = float32(tau*float64(ss[j]) + omt*float64(ds[j]))
				}
			})
			continue
		}
		n := s.Numel()
		for j := 0; j < n; j++ {
			idx := tensor.Unravel(j, s.Shape())
			ov, tv := s.AtF64(idx...), d.AtF64(idx...)
			d.SetF64(tau*ov+(1-tau)*tv, idx...)
		}
	}
}

// clampToBounds clamps action in place into [low,high] (final numerical guard so
// an emitted action is always inside the env's box) and returns it.
func clampToBounds(action, low, high []float64) []float64 {
	for i := range action {
		if action[i] < low[i] {
			action[i] = low[i]
		} else if action[i] > high[i] {
			action[i] = high[i]
		}
	}
	return action
}
