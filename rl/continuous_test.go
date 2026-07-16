package rl

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// cenvClose asserts a≈b within tol.
func cenvClose(t *testing.T, got, want, tol float64, msg string) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v want %v (|Δ|=%g > %g)", msg, got, want, math.Abs(got-want), tol)
	}
}

// TestContinuousEnvPointMass checks the ContinuousEnv contract on PointMass: the
// advertised dims/bounds, deterministic seeded reset, in-box action clamping, and
// the hand-computable reward at the origin.
func TestContinuousEnvPointMass(t *testing.T) {
	var env ContinuousEnv = NewPointMass(2, 1)
	if env.ObsDim() != 4 {
		t.Errorf("ObsDim = %d, want 4", env.ObsDim())
	}
	if env.ActionDim() != 2 {
		t.Errorf("ActionDim = %d, want 2", env.ActionDim())
	}
	low, high := env.ActionBounds()
	if len(low) != 2 || len(high) != 2 {
		t.Fatalf("bounds length %d/%d, want 2/2", len(low), len(high))
	}
	for i := range low {
		cenvClose(t, low[i], -1, 0, "low")
		cenvClose(t, high[i], 1, 0, "high")
	}

	// Deterministic reset: same seed → identical initial observation.
	a := NewPointMass(2, 42).Reset()
	b := NewPointMass(2, 42).Reset()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("reset not deterministic at %d: %v vs %v", i, a[i], b[i])
		}
	}

	// Reward at the origin with zero velocity/action is exactly 0; a huge action is
	// clamped to the ForceMax box (so the control cost saturates, not explodes).
	p := NewPointMass(1, 7)
	p.pos[0], p.vel[0] = 0, 0
	p.steps = 0
	// A zero action from rest keeps us at rest: reward 0 (dist 0, vel 0, act 0).
	obs, r, done := p.Step([]float64{0})
	cenvClose(t, r, 0, 1e-12, "origin reward")
	if done {
		t.Error("episode ended after one step (MaxSteps=40)")
	}
	if len(obs) != 2 {
		t.Errorf("obs len %d, want 2", len(obs))
	}
	// An out-of-box action is clamped to ForceMax=1: velocity becomes 1*Dt=0.1.
	p2 := NewPointMass(1, 7)
	p2.pos[0], p2.vel[0], p2.steps = 0, 0, 0
	p2.Step([]float64{1000})
	cenvClose(t, p2.vel[0], 1*p2.Dt, 1e-12, "clamped velocity")
}

// TestReplayBufferFIFO verifies the ring buffer: capacity is a hard cap, the oldest
// entry is overwritten once full, len saturates at capacity, and sampling is
// deterministic given the seed (§V16 replay anchor).
func TestReplayBufferFIFO(t *testing.T) {
	buf := newReplayBuffer(3, rand.New(rand.NewPCG(1, 2)))
	if buf.len() != 0 {
		t.Fatalf("empty len = %d, want 0", buf.len())
	}
	add := func(id float64) {
		buf.add([]float64{id}, []float64{id}, id, []float64{id}, false)
	}
	add(0)
	add(1)
	if buf.len() != 2 {
		t.Fatalf("len after 2 adds = %d, want 2", buf.len())
	}
	add(2) // now full
	add(3) // overwrites entry 0 (FIFO)
	add(4) // overwrites entry 1
	if buf.len() != 3 {
		t.Fatalf("len = %d, want 3 (capacity cap)", buf.len())
	}
	// The three surviving rewards are the most recent: {2,3,4}. Entry 0 and 1 gone.
	seen := map[float64]bool{}
	for _, tr := range buf.data {
		seen[tr.r] = true
	}
	for _, want := range []float64{2, 3, 4} {
		if !seen[want] {
			t.Errorf("expected reward %v to survive, buffer = %v", want, buf.data)
		}
	}
	for _, gone := range []float64{0, 1} {
		if seen[gone] {
			t.Errorf("expected reward %v to be overwritten", gone)
		}
	}

	// Deterministic sampling: two buffers with the same seed sample identically.
	mk := func() *replayBuffer {
		b := newReplayBuffer(10, rand.New(rand.NewPCG(99, 0)))
		for i := 0; i < 8; i++ {
			b.add([]float64{float64(i)}, nil, float64(i), nil, false)
		}
		return b
	}
	s1 := mk().sample(20)
	s2 := mk().sample(20)
	for i := range s1 {
		if s1[i].r != s2[i].r {
			t.Fatalf("sampling not deterministic at %d: %v vs %v", i, s1[i].r, s2[i].r)
		}
	}
}

// TestSoftUpdateMath verifies the Polyak soft update exactly: target ← τ·online +
// (1−τ)·target elementwise, and τ=1 is an exact hard copy (§V16 soft-update anchor).
func TestSoftUpdateMath(t *testing.T) {
	online := contMLP(3, 5, 2, 11)
	target := contMLP(3, 5, 2, 22) // different init → params differ

	// Snapshot both parameter sets.
	snap := func(net interface{ Params() []*tensor.Tensor }) [][]float64 {
		ps := net.Params()
		out := make([][]float64, len(ps))
		for i, p := range ps {
			out[i] = make([]float64, p.Numel())
			for j := range out[i] {
				out[i][j] = p.AtF64(tensor.Unravel(j, p.Shape())...)
			}
		}
		return out
	}
	on0 := snap(online)
	tg0 := snap(target)

	const tau = 0.05
	softUpdate(online, target, tau)
	tg1 := snap(target)
	for i := range tg1 {
		for j := range tg1[i] {
			want := tau*on0[i][j] + (1-tau)*tg0[i][j]
			cenvClose(t, tg1[i][j], want, 1e-12, "soft-update entry")
		}
	}
	// Online must be untouched.
	on1 := snap(online)
	for i := range on1 {
		for j := range on1[i] {
			cenvClose(t, on1[i][j], on0[i][j], 0, "online unchanged")
		}
	}

	// τ=1 is an exact hard copy: target becomes online.
	softUpdate(online, target, 1)
	tg2 := snap(target)
	for i := range tg2 {
		for j := range tg2[i] {
			cenvClose(t, tg2[i][j], on0[i][j], 1e-12, "hard-copy (τ=1)")
		}
	}
}

// evalPolicy runs one evaluation episode on env with the given policy and returns
// the undiscounted return.
func evalPolicy(env ContinuousEnv, policy func(obs []float64) []float64) float64 {
	obs := env.Reset()
	var total float64
	for {
		next, r, done := env.Step(policy(obs))
		total += r
		obs = next
		if done {
			return total
		}
	}
}
