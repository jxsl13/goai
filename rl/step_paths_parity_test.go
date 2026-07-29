package rl

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// digest reduces a float slice to two order-sensitive-in-one, order-independent-in-the-other
// words: the bitwise running sum catches a changed magnitude, the xor catches any single bit
// that moved even if the sum happens to cancel. Both are needed — a reversed summation order
// leaves the sum bit-pattern intact while moving the xor.
func digest(vs ...[]float64) (uint64, uint64) {
	var s float64
	var xr uint64
	for _, v := range vs {
		for _, f := range v {
			s += f
			xr ^= math.Float64bits(f)
		}
	}
	return math.Float64bits(s), xr
}

// TestRolloutTrajectoryParity pins the whole collected trajectory at a fixed seed.
//
// The thing at risk when the critic call moves out of the collection loop is not the
// critic's arithmetic but the RNG STREAM: the actor forward, softmax and sampleAction must
// stay in place and in order, or every action after the first divergence changes. Actions,
// rewards and dones are therefore pinned alongside the values. Separately, ro.values moves
// from 256 batch-1 forwards to one batch-256 forward, which is only safe if the GEMV and
// band kernels agree at tolerance 0 on this host — pinning the values digest is what
// asserts that rather than assuming it.
func TestRolloutTrajectoryParity(t *testing.T) {
	env := NewChain(9, 50)
	actor := policyNet(env.ObsDim(), 64, env.NumActions(), 1)
	critic := nn.NewSequential(
		nn.NewLinear(tensor.F64, env.ObsDim(), 64, 2),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, 64, 1, 3),
	)
	ro, err := rlRollout(env, actor, critic, rand.New(rand.NewPCG(4, 5)), 256)
	if err != nil {
		t.Fatal(err)
	}
	acts := make([]float64, len(ro.actions))
	for i, a := range ro.actions {
		acts[i] = float64(a)
	}
	dones := make([]float64, len(ro.dones))
	for i, d := range ro.dones {
		if d {
			dones[i] = 1
		}
	}
	// RNG stream and env interaction: must be bit-for-bit untouched.
	const wantTrajSum, wantTrajXor uint64 = 0xc0313bd1948faa70, 0x3ff727df17ea2494
	if sum, xor := digest(acts, ro.rewards, dones, ro.logpOld); sum != wantTrajSum || xor != wantTrajXor {
		t.Errorf("trajectory digest = %016x/%016x, want %016x/%016x — the RNG stream moved",
			sum, xor, wantTrajSum, wantTrajXor)
	}
	if len(ro.states) != 270 {
		t.Errorf("collected %d steps, want 270", len(ro.states))
	}
	// Critic values: the only quantity batching can legitimately move, and only if the m=1
	// and m=N kernels disagree. They agree at tolerance 0 on darwin/arm64 (both reach the
	// scalar band kernel); this assertion is what establishes that rather than assuming it,
	// and it is the one to look at first if this test ever fails on another host.
	const wantValSum, wantValXor uint64 = 0x401cbd8fffab3a70, 0x8064face2ad9b4a1
	if sum, xor := digest(ro.values); sum != wantValSum || xor != wantValXor {
		t.Errorf("values digest = %016x/%016x, want %016x/%016x — m=1 vs m=N kernel disagreement",
			sum, xor, wantValSum, wantValXor)
	}
}

// TestDQNLearnParity pins the online-net parameters after a fixed number of seeded learn
// steps. Removing the redundant forward must not change any value or consume a different
// number of random draws.
func TestDQNLearnParity(t *testing.T) {
	env := NewChain(9, 50)
	d := NewDQN(env, 64, 1e-3, 0.99, 11)
	d.BatchSize = 32
	obs := env.Reset()
	for range 512 {
		a := d.rng.IntN(env.NumActions())
		next, rew, done := env.Step(a)
		d.buf = append(d.buf, transition{s: obs, s2: next, a: a, r: rew, done: done})
		obs = next
		if done {
			obs = env.Reset()
		}
	}
	k := env.NumActions()
	for range 20 {
		if err := d.learn(k); err != nil {
			t.Fatal(err)
		}
	}
	var ps [][]float64
	for _, p := range d.Net.Params() {
		ps = append(ps, p.Storage().F64())
	}
	const wantSum, wantXor uint64 = 0xc029655a6286c8b7, 0xbf25921a602a9c5a
	if sum, xor := digest(ps...); sum != wantSum || xor != wantXor {
		t.Errorf("params digest = %016x/%016x, want %016x/%016x", sum, xor, wantSum, wantXor)
	}
}
