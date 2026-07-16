package rl

// Degenerate-input robustness (hostile) audit for the RL agents (PPO, A2C,
// Q-learning, SARSA). Every test drives an adversarial-but-VALID degenerate
// setting — a single-action env, a zero-reward or all-equal-reward rollout, a
// one-step episode, extreme ±1e6 rewards, and a near-deterministic (collapsed)
// policy — and asserts finite loss, finite gradients, finite learned tables, and
// no panic. Helpers are prefixed rlhostile* to stay clear of the ppo*/a2c*/
// tabular*/hostile* names already in use.
//
// Bug found + fixed by this audit: rlNegEntropy computed the entropy term as
// softmax·log(softmax); for a near-deterministic policy a non-max softmax entry
// underflows to exactly 0, so log(0) = −∞ made the 0·log0 term 0·(−∞) = NaN in
// BOTH the forward value and (via 1/p in log's VJP) the gradient. The fix uses a
// stable log-softmax (always finite for finite logits). TestRLHostileEntropyNear
// Deterministic is the regression guard.

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/tensor"
)

// rlhostileEnv is a configurable degenerate episodic environment: a fixed number
// of actions and one-hot observation slots, a constant per-step reward, and a
// hard step cap (maxSteps=1 gives a one-step episode). The observation is always
// the same one-hot vector, so the policy sees a single effective state.
type rlhostileEnv struct {
	nActions, obsDim, maxSteps int
	reward                     float64
	step                       int
}

func (e *rlhostileEnv) NumActions() int { return e.nActions }
func (e *rlhostileEnv) ObsDim() int     { return e.obsDim }
func (e *rlhostileEnv) Reset() []float64 {
	e.step = 0
	o := make([]float64, e.obsDim)
	o[0] = 1
	return o
}
func (e *rlhostileEnv) Step(int) ([]float64, float64, bool) {
	e.step++
	o := make([]float64, e.obsDim)
	o[0] = 1
	return o, e.reward, e.step >= e.maxSteps
}

// rlhostileFinite fails the test if any element of t is NaN or ±Inf.
func rlhostileFinite(t *testing.T, name string, x *tensor.Tensor) {
	t.Helper()
	if x == nil {
		return
	}
	for i := 0; i < x.Numel(); i++ {
		v := x.AtF64(tensor.Unravel(i, x.Shape())...)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s: non-finite value %v at flat index %d", name, v, i)
		}
	}
}

// rlhostileParamsFinite asserts every trainable parameter is finite.
func rlhostileParamsFinite(t *testing.T, name string, ps []*tensor.Tensor) {
	t.Helper()
	for i, p := range ps {
		rlhostileFinite(t, name+"/param", p)
		_ = i
	}
}

// rlhostileCases enumerates the degenerate rollout settings shared by the PPO and
// A2C update tests.
func rlhostileCases() []struct {
	name                       string
	nActions, obsDim, maxSteps int
	reward                     float64
} {
	return []struct {
		name                       string
		nActions, obsDim, maxSteps int
		reward                     float64
	}{
		{"single-action", 1, 4, 8, 1.0},    // NumActions=1: softmax over 1 = 1.0, logπ=0
		{"zero-reward", 3, 4, 8, 0.0},      // reward everywhere 0
		{"all-equal-reward", 3, 4, 8, 2.5}, // constant reward → advantage std 0
		{"extreme+1e6", 3, 4, 8, 1e6},      // extreme positive reward
		{"extreme-1e6", 3, 4, 8, -1e6},     // extreme negative reward
		{"one-step", 3, 4, 1, 1.0},         // episode terminates after a single step
	}
}

// TestRLHostileEntropyNearDeterministic is the regression guard for the fixed
// 0·log0 entropy NaN: at a near-deterministic policy (huge logit gaps that
// underflow the small-probability softmax entries to exactly 0) the negative
// entropy and its gradient must both be finite (the x·logx → 0 limit).
func TestRLHostileEntropyNearDeterministic(t *testing.T) {
	for _, gap := range []float64{50, 400, 1000, 1e6} {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits := tensor.New(tensor.F64, tensor.Shape{2, 3})
		// row 0: one dominant action; row 1: two near-tied dominant, one crushed.
		logits.SetF64(gap, 0, 0)
		logits.SetF64(0, 0, 1)
		logits.SetF64(-gap, 0, 2)
		logits.SetF64(gap, 1, 0)
		logits.SetF64(gap, 1, 1)
		logits.SetF64(-gap, 1, 2)
		negH, err := rlNegEntropy(ctx, logits)
		if err != nil {
			t.Fatalf("gap %g: rlNegEntropy: %v", gap, err)
		}
		rlhostileFinite(t, "negEntropy", negH)
		if err := tape.Backward(negH); err != nil {
			t.Fatalf("gap %g: backward: %v", gap, err)
		}
		rlhostileFinite(t, "negEntropy/grad", tape.Grad(logits))
	}
}

// TestRLHostilePPOUpdateDegenerate runs several PPO updates over each degenerate
// env and asserts the actor/critic parameters stay finite with no panic or error.
func TestRLHostilePPOUpdateDegenerate(t *testing.T) {
	for _, c := range rlhostileCases() {
		t.Run(c.name, func(t *testing.T) {
			env := &rlhostileEnv{nActions: c.nActions, obsDim: c.obsDim, maxSteps: c.maxSteps, reward: c.reward}
			ag := NewPPO(env, 8, 0.05, 0.99, 1, WithPPORolloutSteps(16))
			for it := 0; it < 5; it++ {
				ret, err := ag.Update(env)
				if err != nil {
					t.Fatalf("iter %d: %v", it, err)
				}
				if math.IsNaN(ret) || math.IsInf(ret, 0) {
					t.Fatalf("iter %d: non-finite mean return %v", it, ret)
				}
				rlhostileParamsFinite(t, "ppo-actor", ag.Actor.Params())
				rlhostileParamsFinite(t, "ppo-critic", ag.Critic.Params())
			}
		})
	}
}

// TestRLHostileA2CUpdateDegenerate is the A2C counterpart of the PPO update test.
func TestRLHostileA2CUpdateDegenerate(t *testing.T) {
	for _, c := range rlhostileCases() {
		t.Run(c.name, func(t *testing.T) {
			env := &rlhostileEnv{nActions: c.nActions, obsDim: c.obsDim, maxSteps: c.maxSteps, reward: c.reward}
			ag := NewA2C(env, 8, 0.05, 0.99, 1, WithA2CRolloutSteps(16))
			for it := 0; it < 5; it++ {
				ret, err := ag.Update(env)
				if err != nil {
					t.Fatalf("iter %d: %v", it, err)
				}
				if math.IsNaN(ret) || math.IsInf(ret, 0) {
					t.Fatalf("iter %d: non-finite mean return %v", it, ret)
				}
				rlhostileParamsFinite(t, "a2c-actor", ag.Actor.Params())
				rlhostileParamsFinite(t, "a2c-critic", ag.Critic.Params())
			}
		})
	}
}

// TestRLHostileNormAdvZeroStd checks the advantage normalization against a
// zero-variance advantage vector (every reward equal → std 0): the ε floor must
// keep the standardized advantages finite (0), not divide by zero.
func TestRLHostileNormAdvZeroStd(t *testing.T) {
	for _, val := range []float64{0, 5, -1e6, 1e6} {
		adv := []float64{val, val, val, val, val}
		out := rlNormAdv(adv)
		rlhostileFinite(t, "normAdv", out)
		if v := out.AtF64(0); v != 0 {
			t.Errorf("val %g: zero-variance advantage normalized to %v, want 0", val, v)
		}
	}
}

// TestRLHostilePPOCollapse covers the reachable near-deterministic-collapse
// directions of the PPO objective. (1) A collapsed actor's rollout never emits a
// non-finite frozen log-prob logπ_old — the sampler draws the dominant action, so
// logπ_old stays finite. (2) When the current policy has since moved AWAY from a
// rollout action (logπθ ≪ logπ_old), the importance ratio underflows to 0 and the
// clipped surrogate and its gradient stay finite.
func TestRLHostilePPOCollapse(t *testing.T) {
	// (1) rollout log-probs finite under a collapsed actor.
	env := &rlhostileEnv{nActions: 3, obsDim: 4, maxSteps: 8, reward: 1.0}
	p := NewPPO(env, 8, 0.05, 0.99, 1, WithPPORolloutSteps(64))
	for _, prm := range p.Actor.Params() {
		for i := 0; i < prm.Numel(); i++ {
			idx := tensor.Unravel(i, prm.Shape())
			prm.SetF64(prm.AtF64(idx...)*1e3, idx...) // blow up logits → collapsed softmax
		}
	}
	ro, err := rlRollout(env, p.Actor, p.Critic, rand.New(rand.NewPCG(1, 2)), 64)
	if err != nil {
		t.Fatal(err)
	}
	for i, lp := range ro.logpOld {
		if math.IsNaN(lp) || math.IsInf(lp, 0) {
			t.Fatalf("collapsed rollout logpOld[%d] = %v (non-finite)", i, lp)
		}
	}

	// (2) ratio underflow direction: current policy strongly prefers action 0,
	// batch took actions 1 and 2 → logπθ ≪ logπ_old → r → 0.
	states := [][]float64{{1, 0, 0, 0}, {1, 0, 0, 0}}
	actions := []int{1, 2}
	net := policyNet(4, 8, 3, 7)
	w := net.Params()[len(net.Params())-2] // last Linear weight [hidden, 3]
	for i := 0; i < w.Shape()[0]; i++ {
		w.SetF64(50, i, 0)
	}
	tape := autograd.NewTape()
	ctx := tape.Context()
	loss, err := ppoPolicyLoss(ctx, net, states, actions, rlVec([]float64{-0.5, -0.5}), rlVec([]float64{1, -1}), 0.2, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	rlhostileFinite(t, "ppo-collapse-loss", loss)
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	for _, prm := range net.Params() {
		rlhostileFinite(t, "ppo-collapse-grad", tape.Grad(prm))
	}
}

// TestRLHostileTabularDegenerate runs Q-learning and SARSA over the degenerate
// envs (single-action, zero/all-equal/extreme rewards, one-step episodes) and
// asserts every Q-table entry stays finite and the extracted policy is valid.
func TestRLHostileTabularDegenerate(t *testing.T) {
	for _, c := range rlhostileCases() {
		t.Run(c.name, func(t *testing.T) {
			envQ := &rlhostileEnv{nActions: c.nActions, obsDim: c.obsDim, maxSteps: c.maxSteps, reward: c.reward}
			envS := &rlhostileEnv{nActions: c.nActions, obsDim: c.obsDim, maxSteps: c.maxSteps, reward: c.reward}
			q := NewQLearning(envQ, 0.5, 0.99, 0.1, 1)
			s := NewSARSA(envS, 0.5, 0.99, 0.1, 1)
			for it := 0; it < 10; it++ {
				if _, err := q.Episode(envQ); err != nil {
					t.Fatalf("Q iter %d: %v", it, err)
				}
				if _, err := s.Episode(envS); err != nil {
					t.Fatalf("SARSA iter %d: %v", it, err)
				}
			}
			for name, ag := range map[string]*tabularAgent{"qlearning": &q.tabularAgent, "sarsa": &s.tabularAgent} {
				for st, row := range ag.QTable() {
					for a, v := range row {
						if math.IsNaN(v) || math.IsInf(v, 0) {
							t.Fatalf("%s Q[%d][%d] = %v (non-finite)", name, st, a, v)
						}
					}
				}
				for st, act := range ag.Policy() {
					if act < 0 || act >= c.nActions {
						t.Fatalf("%s policy[%d] = %d out of [0,%d)", name, st, act, c.nActions)
					}
				}
			}
		})
	}
}
