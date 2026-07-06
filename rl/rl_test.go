package rl

import (
	"math"
	"testing"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Exactly verifiable piece: discounted returns against hand-computed values.
func TestDiscountedReturns(t *testing.T) {
	g := DiscountedReturns([]float64{1, 0, 2}, 0.5)
	// G2 = 2; G1 = 0 + 0.5·2 = 1; G0 = 1 + 0.5·1 = 1.5
	want := []float64{1.5, 1, 2}
	for i := range want {
		if math.Abs(g[i]-want[i]) > 1e-15 {
			t.Errorf("G[%d] = %v, want %v", i, g[i], want[i])
		}
	}
	if got := DiscountedReturns(nil, 0.9); len(got) != 0 {
		t.Error("empty rewards → empty returns")
	}
}

func TestChainEnv(t *testing.T) {
	env := NewChain(5, 20)
	obs := env.Reset()
	if len(obs) != 5 || obs[2] != 1 {
		t.Fatalf("reset obs %v, want one-hot at 2", obs)
	}
	// two rights from middle → terminal with reward 1
	_, r, done := env.Step(1)
	if done || r != 0 {
		t.Fatal("one right from middle must not terminate")
	}
	_, r, done = env.Step(1)
	if !done || r != 1.0 {
		t.Fatalf("right end: r=%v done=%v", r, done)
	}
	// left end pays 0.1
	env.Reset()
	env.Step(0)
	_, r, done = env.Step(0)
	if !done || r != 0.1 {
		t.Fatalf("left end: r=%v done=%v", r, done)
	}
}

// §T26: REINFORCE learns "go right" — late-episode average reward ≥ 0.9 and
// clearly better than the early phase. Deterministic seeds.
func TestReinforceLearnsChain(t *testing.T) {
	env := NewChain(5, 20)
	agent := NewReinforce(env, 16, 0.05, 0.99, 42)

	const episodes = 300
	var early, late float64
	for ep := range episodes {
		rew, err := agent.Episode(env)
		if err != nil {
			t.Fatal(err)
		}
		if ep < 50 {
			early += rew
		}
		if ep >= episodes-50 {
			late += rew
		}
	}
	early /= 50
	late /= 50
	if late < 0.9 {
		t.Fatalf("REINFORCE did not learn: late avg %.3f (early %.3f)", late, early)
	}
	if late <= early {
		t.Fatalf("no improvement: early %.3f → late %.3f", early, late)
	}
	t.Logf("REINFORCE: early avg %.3f → late avg %.3f", early, late)
}

// §T26: DQN learns the optimal greedy policy (right from every interior state)
// and Q(s,right) > Q(s,left) — the Bellman-consistent ordering.
func TestDQNLearnsChain(t *testing.T) {
	env := NewChain(5, 20)
	agent := NewDQN(env, 16, 0.01, 0.9, 7)

	var late float64
	const episodes = 400
	for ep := range episodes {
		rew, err := agent.Episode(env)
		if err != nil {
			t.Fatal(err)
		}
		if ep >= episodes-50 {
			late += rew
		}
	}
	late /= 50
	t.Logf("DQN late avg reward %.3f (eps %.3f)", late, agent.Eps)

	// greedy policy: right from every interior state; Q-ordering holds
	for pos := 1; pos < 4; pos++ {
		obs := make([]float64, 5)
		obs[pos] = 1
		a, err := agent.GreedyAction(obs, 2)
		if err != nil {
			t.Fatal(err)
		}
		if a != 1 {
			q, _ := agent.QValues(obs, 2)
			t.Fatalf("state %d: greedy action %d (Q=%v), want right", pos, a, q)
		}
		q, err := agent.QValues(obs, 2)
		if err != nil {
			t.Fatal(err)
		}
		if q[1] <= q[0] {
			t.Fatalf("state %d: Q(right)=%v ≤ Q(left)=%v", pos, q[1], q[0])
		}
	}
}
