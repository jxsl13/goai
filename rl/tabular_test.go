package rl_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/rl"
)

// ---------------------------------------------------------------------------
// Examples (apicheck: runnable ExampleQLearning/ExampleSARSA with //Output).
// ---------------------------------------------------------------------------

// Q-learning solves the Chain by trial and error: after training its extracted
// greedy policy is "always right" (action 1) at every interior state, the optimal
// policy. Deterministic via the seed.
func ExampleQLearning() {
	env := rl.NewChain(5 /*states*/, 20 /*maxSteps*/)
	agent := rl.NewQLearning(env, 0.5 /*α*/, 0.9 /*γ*/, 0.2 /*ε*/, 1 /*seed*/)
	for range 1500 {
		if _, err := agent.Episode(env); err != nil {
			panic(err)
		}
	}
	pol := agent.Policy()                                // greedy action per state
	fmt.Println("policy[1..3]:", pol[1], pol[2], pol[3]) // interior states → right
	fmt.Println("greedy@start:", agent.GreedyAction(2))  // 1 = right
	fmt.Println("states:", len(agent.QTable()))          // one Q-row per state
	// Output:
	// policy[1..3]: 1 1 1
	// greedy@start: 1
	// states: 5
}

// SARSA (on-policy TD control) also learns "always right" on the Chain, where the
// optimal path carries no exploration risk.
func ExampleSARSA() {
	env := rl.NewChain(5, 20)
	agent := rl.NewSARSA(env, 0.5, 0.9, 0.1, 1)
	for range 500 {
		if _, err := agent.Episode(env); err != nil {
			panic(err)
		}
	}
	fmt.Println("greedy@start:", agent.GreedyAction(2))
	// Output:
	// greedy@start: 1
}

// ---------------------------------------------------------------------------
// Anchor 1: converges to the optimal policy on Chain (both agents).
// ---------------------------------------------------------------------------

// tabularGreedyReward rolls out the extracted greedy policy (ε set to 0) and
// returns the episode reward — the achievable value of the learned policy.
func tabularGreedyReward(t *testing.T, greedy func(int) int, env rl.Env) float64 {
	t.Helper()
	obs := env.Reset()
	var total float64
	for i := 0; i < 1000; i++ {
		a := greedy(argmaxOneHot(obs))
		next, r, done := env.Step(a)
		total += r
		obs = next
		if done {
			return total
		}
	}
	return total
}

func argmaxOneHot(obs []float64) int {
	best, bv := 0, obs[0]
	for i, v := range obs {
		if v > bv {
			bv, best = v, i
		}
	}
	return best
}

func TestTabularConvergesToOptimalPolicy(t *testing.T) {
	const episodes = 1500
	for _, tc := range []struct {
		name string
		make func(rl.Env) interface {
			Episode(rl.Env) (float64, error)
			GreedyAction(int) int
			Policy() []int
		}
	}{
		{"QLearning", func(e rl.Env) interface {
			Episode(rl.Env) (float64, error)
			GreedyAction(int) int
			Policy() []int
		} {
			return rl.NewQLearning(e, 0.5, 0.9, 0.2, 3)
		}},
		{"SARSA", func(e rl.Env) interface {
			Episode(rl.Env) (float64, error)
			GreedyAction(int) int
			Policy() []int
		} {
			return rl.NewSARSA(e, 0.5, 0.9, 0.2, 3)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			env := rl.NewChain(5, 20)
			agent := tc.make(env)
			var early, late float64
			for ep := 0; ep < episodes; ep++ {
				r, err := agent.Episode(env)
				if err != nil {
					t.Fatal(err)
				}
				if ep < 50 {
					early += r
				}
				if ep >= episodes-50 {
					late += r
				}
			}
			early /= 50
			late /= 50
			t.Logf("%s learning curve: early avg %.3f → late avg %.3f (online, ε=0.2)", tc.name, early, late)

			// extracted greedy policy must be optimal: right at every interior state.
			pol := agent.Policy()
			for s := 1; s <= 3; s++ {
				if pol[s] != 1 {
					t.Fatalf("%s greedy policy at interior state %d = %d, want 1 (right)", tc.name, s, pol[s])
				}
			}
			// greedy rollout must reach the achievable optimum reward (1.0).
			if got := tabularGreedyReward(t, agent.GreedyAction, rl.NewChain(5, 20)); math.Abs(got-1.0) > 1e-9 {
				t.Fatalf("%s greedy rollout reward = %.3f, want 1.0 (optimal)", tc.name, got)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Anchor 2: Bellman optimality — learned Q ≈ analytic Q* on Chain.
// ---------------------------------------------------------------------------

// analyticChainQStar computes the exact optimal action-value function for the
// deterministic Chain(n) via value iteration (converges exactly for a finite
// deterministic MDP). Actions: 0=left, 1=right; terminal states 0 and n-1.
func analyticChainQStar(n int, gamma float64) [][]float64 {
	q := make([][]float64, n)
	for i := range q {
		q[i] = make([]float64, 2)
	}
	for iter := 0; iter < 2000; iter++ {
		for s := 1; s < n-1; s++ {
			for a := 0; a < 2; a++ {
				ns := s - 1
				if a == 1 {
					ns = s + 1
				}
				r := 0.0
				done := false
				switch {
				case ns >= n-1:
					r, done = 1.0, true
				case ns <= 0:
					r, done = 0.1, true
				}
				t := r
				if !done {
					t += gamma * math.Max(q[ns][0], q[ns][1])
				}
				q[s][a] = t
			}
		}
	}
	return q
}

func TestTabularBellmanOptimality(t *testing.T) {
	const gamma = 0.9
	env := rl.NewChain(5, 20)
	// α=1 exact bootstrap on a deterministic env; ε=0.2 to visit every (s,a).
	agent := rl.NewQLearning(env, 1.0, gamma, 0.2, 11)
	for ep := 0; ep < 4000; ep++ {
		if _, err := agent.Episode(env); err != nil {
			t.Fatal(err)
		}
	}
	qstar := analyticChainQStar(5, gamma)
	q := agent.QTable()
	for s := 1; s <= 3; s++ {
		for a := 0; a < 2; a++ {
			if d := math.Abs(q[s][a] - qstar[s][a]); d > 0.02 {
				t.Fatalf("Q[%d][%d] = %.4f, analytic Q* = %.4f (Δ=%.4f > tol)", s, a, q[s][a], qstar[s][a], d)
			}
		}
	}
	// self-consistency: the learned Q also satisfies the Bellman optimality
	// equation Q*(s,a) = r + γ·max_a' Q*(s',a') directly.
	// state 3, right → terminal right end, reward 1.0 (no bootstrap).
	if d := math.Abs(q[3][1] - 1.0); d > 0.02 {
		t.Fatalf("Bellman: Q[3][right] = %.4f, want ≈1.0", q[3][1])
	}
	// state 2, right → state 3, reward 0, bootstrap γ·max Q[3].
	want := gamma * math.Max(q[3][0], q[3][1])
	if d := math.Abs(q[2][1] - want); d > 0.02 {
		t.Fatalf("Bellman: Q[2][right] = %.4f, want γ·maxQ(3) = %.4f", q[2][1], want)
	}
	t.Logf("Bellman-optimal: learned Q[1..3] within 0.02 of analytic Q* (γ=%.2f)", gamma)
}

// ---------------------------------------------------------------------------
// Anchor 3: Q-learning vs SARSA cliff-walking distinction.
// ---------------------------------------------------------------------------

// tabularCliff is a Sutton-Barto cliff-walking grid: rows×cols, start at the
// bottom-left, goal at the bottom-right, and the bottom-row interior is a cliff.
// Each step costs −1; stepping onto the cliff costs −100 and teleports back to the
// start (episode continues). One-hot observations over rows·cols states, 4 actions
// (0=up, 1=down, 2=left, 3=right). The optimal path hugs the cliff edge; under
// ε-greedy exploration that edge is risky, so on-policy SARSA prefers a safer route.
type tabularCliff struct {
	rows, cols, max int
	pos, steps      int
}

func newTabularCliff(rows, cols, max int) *tabularCliff {
	return &tabularCliff{rows: rows, cols: cols, max: max}
}
func (c *tabularCliff) NumActions() int { return 4 }
func (c *tabularCliff) ObsDim() int     { return c.rows * c.cols }
func (c *tabularCliff) startPos() int   { return (c.rows - 1) * c.cols }
func (c *tabularCliff) goalPos() int    { return (c.rows-1)*c.cols + c.cols - 1 }
func (c *tabularCliff) obs() []float64 {
	o := make([]float64, c.rows*c.cols)
	o[c.pos] = 1
	return o
}
func (c *tabularCliff) Reset() []float64 {
	c.pos, c.steps = c.startPos(), 0
	return c.obs()
}
func (c *tabularCliff) Step(action int) ([]float64, float64, bool) {
	c.steps++
	r, col := c.pos/c.cols, c.pos%c.cols
	switch action {
	case 0: // up
		if r > 0 {
			r--
		}
	case 1: // down
		if r < c.rows-1 {
			r++
		}
	case 2: // left
		if col > 0 {
			col--
		}
	case 3: // right
		if col < c.cols-1 {
			col++
		}
	}
	c.pos = r*c.cols + col
	capped := c.steps >= c.max
	if c.pos == c.goalPos() {
		return c.obs(), -1, true
	}
	if r == c.rows-1 && col >= 1 && col <= c.cols-2 { // cliff
		c.pos = c.startPos()
		return c.obs(), -100, capped
	}
	return c.obs(), -1, capped
}

// tabularCliffGreedyPath rolls out the greedy policy and returns the visited state
// indices and path length. cliffApproach reports the largest row reached in a
// cliff-adjacent column (columns 1..cols-2) — a proxy for how close the learned
// route runs to the cliff (larger = closer to the bottom-row cliff = riskier).
func tabularCliffGreedyPath(greedy func(int) int, c *tabularCliff) (states []int, length int, reached bool) {
	obs := c.Reset()
	for i := 0; i < 500; i++ {
		s := argmaxOneHot(obs)
		states = append(states, s)
		next, _, done := c.Step(greedy(s))
		length++
		obs = next
		if c.pos == c.goalPos() {
			states = append(states, c.pos)
			return states, length, true
		}
		if done {
			return states, length, false
		}
	}
	return states, length, false
}

func cliffApproach(states []int, cols int) int {
	maxRow := 0
	for _, s := range states {
		r, col := s/cols, s%cols
		if col >= 1 && col <= cols-2 && r > maxRow {
			maxRow = r
		}
	}
	return maxRow
}

func TestTabularQLearningVsSARSACliff(t *testing.T) {
	// 4 rows × 12 cols (the Sutton-Barto layout): row 3 = start/cliff/goal, row 2 =
	// the cliff edge, rows 0–1 offer a safe detour. Undiscounted (γ=1), ε=0.1.
	const rows, cols = 4, 12
	newEnv := func() *tabularCliff { return newTabularCliff(rows, cols, 400) }
	const episodes = 2000

	ql := rl.NewQLearning(newEnv(), 0.5, 1.0, 0.1, 5)
	sa := rl.NewSARSA(newEnv(), 0.5, 1.0, 0.1, 5)
	var qlOnline, saOnline float64
	const window = 300
	envQ, envS := newEnv(), newEnv()
	for ep := 0; ep < episodes; ep++ {
		rq, err := ql.Episode(envQ)
		if err != nil {
			t.Fatal(err)
		}
		rs, err := sa.Episode(envS)
		if err != nil {
			t.Fatal(err)
		}
		if ep >= episodes-window {
			qlOnline += rq
			saOnline += rs
		}
	}
	qlOnline /= window
	saOnline /= window

	qStates, qLen, qReached := tabularCliffGreedyPath(ql.GreedyAction, newEnv())
	sStates, sLen, sReached := tabularCliffGreedyPath(sa.GreedyAction, newEnv())
	if !qReached || !sReached {
		t.Fatalf("greedy policies must reach goal: q=%v s=%v", qReached, sReached)
	}
	qApproach := cliffApproach(qStates, cols) // expect row 2 (the edge)
	sApproach := cliffApproach(sStates, cols) // expect row 0/1 (safe detour)
	t.Logf("Q-learning: greedy path len %d, cliff-column max row %d, online avg %.1f", qLen, qApproach, qlOnline)
	t.Logf("SARSA:      greedy path len %d, cliff-column max row %d, online avg %.1f", sLen, sApproach, saOnline)

	// Off-policy Q-learning learns the optimal RISKY edge path (shorter, hugs the
	// cliff edge); on-policy SARSA learns the SAFER path (a longer top-row detour).
	// The classic Sutton-Barto cliff-walking result.
	if qLen >= sLen {
		t.Fatalf("expected Q-learning greedy path (%d) shorter than SARSA (%d)", qLen, sLen)
	}
	if qApproach <= sApproach {
		t.Fatalf("expected Q-learning path to hug the cliff (cliff-column row %d) closer than SARSA (%d)", qApproach, sApproach)
	}
	// on-policy: SARSA avoids cliff falls during ε-greedy behaviour, so its online
	// average return is clearly higher (less negative) than Q-learning's.
	if saOnline <= qlOnline {
		t.Fatalf("expected SARSA online return (%.1f) > Q-learning (%.1f) under exploration", saOnline, qlOnline)
	}
}

// ---------------------------------------------------------------------------
// Anchor 4: α=0 (no learning) and γ=0 (myopic → immediate reward).
// ---------------------------------------------------------------------------

func TestTabularAlphaZeroNoLearning(t *testing.T) {
	env := rl.NewChain(5, 20)
	agent := rl.NewQLearning(env, 0.0, 0.9, 0.5, 9)
	for ep := 0; ep < 200; ep++ {
		if _, err := agent.Episode(env); err != nil {
			t.Fatal(err)
		}
	}
	for s, row := range agent.QTable() {
		for a, v := range row {
			if v != 0 {
				t.Fatalf("α=0 must leave Q at init: Q[%d][%d] = %v, want 0", s, a, v)
			}
		}
	}
}

func TestTabularGammaZeroMyopic(t *testing.T) {
	env := rl.NewChain(5, 20)
	// γ=0, α=1 on a deterministic env ⇒ Q(s,a) = immediate reward r(s,a) exactly.
	agent := rl.NewQLearning(env, 1.0, 0.0, 0.3, 13)
	for ep := 0; ep < 3000; ep++ {
		if _, err := agent.Episode(env); err != nil {
			t.Fatal(err)
		}
	}
	q := agent.QTable()
	// state 3 right → right end (reward 1.0); state 1 left → left end (reward 0.1);
	// all interior transitions that don't terminate pay 0.
	cases := []struct {
		s, a int
		want float64
	}{
		{3, 1, 1.0}, // right end
		{1, 0, 0.1}, // left end
		{2, 1, 0.0}, // interior step
		{2, 0, 0.0}, // interior step
	}
	for _, c := range cases {
		if d := math.Abs(q[c.s][c.a] - c.want); d > 1e-6 {
			t.Fatalf("γ=0 myopic: Q[%d][%d] = %.6f, want immediate reward %.3f", c.s, c.a, q[c.s][c.a], c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Anchor 5: ε-greedy correctness (ε=0 pure greedy; ε=1 uniform; decay schedule).
// ---------------------------------------------------------------------------

// tabularBandit is a one-state, k-action environment that records the action taken
// and terminates immediately — used to observe the ε-greedy action distribution.
type tabularBandit struct {
	k    int
	last int
}

func (b *tabularBandit) NumActions() int  { return b.k }
func (b *tabularBandit) ObsDim() int      { return 1 }
func (b *tabularBandit) Reset() []float64 { return []float64{1} }
func (b *tabularBandit) Step(a int) ([]float64, float64, bool) {
	b.last = a
	return []float64{1}, 0, true
}

func TestTabularEpsilonGreedy(t *testing.T) {
	// ε=0 ⇒ pure greedy: with an all-zero Q-table the greedy action is the
	// lowest-index tie-break (0), deterministically, every episode.
	t.Run("epsilon0_pure_greedy", func(t *testing.T) {
		env := &tabularBandit{k: 3}
		// α=0 so Q never changes → greedy stays deterministic.
		agent := rl.NewQLearning(env, 0.0, 0.9, 0.0, 1)
		for i := 0; i < 200; i++ {
			if _, err := agent.Episode(env); err != nil {
				t.Fatal(err)
			}
			if env.last != 0 {
				t.Fatalf("ε=0 pure greedy must always pick argmax (0), got %d", env.last)
			}
		}
	})

	// ε=1 ⇒ uniform random over actions, independent of Q.
	t.Run("epsilon1_uniform", func(t *testing.T) {
		env := &tabularBandit{k: 4}
		agent := rl.NewQLearning(env, 0.0, 0.9, 1.0, 2)
		counts := make([]int, 4)
		const n = 8000
		for i := 0; i < n; i++ {
			if _, err := agent.Episode(env); err != nil {
				t.Fatal(err)
			}
			counts[env.last]++
		}
		exp := float64(n) / 4
		for a, c := range counts {
			if math.Abs(float64(c)-exp)/exp > 0.1 {
				t.Fatalf("ε=1 action %d count %d deviates >10%% from uniform %.0f", a, c, exp)
			}
		}
		t.Logf("ε=1 action counts: %v (uniform ≈ %.0f each)", counts, exp)
	})

	// decay schedule: ε ← max(εMin, ε·decay) per episode.
	t.Run("decay_schedule", func(t *testing.T) {
		env := rl.NewChain(5, 20)
		agent := rl.NewQLearning(env, 0.1, 0.9, 1.0, 4, rl.WithEpsilonDecay(0.5, 0.05))
		want := 1.0
		for ep := 0; ep < 6; ep++ {
			if _, err := agent.Episode(env); err != nil {
				t.Fatal(err)
			}
			want *= 0.5
			if want < 0.05 {
				want = 0.05
			}
			if math.Abs(agent.Epsilon-want) > 1e-9 {
				t.Fatalf("after %d episodes ε = %.5f, want %.5f", ep+1, agent.Epsilon, want)
			}
		}
	})
}

// ---------------------------------------------------------------------------
// Anchor 6: sanity — non-one-hot obs errors clearly; determinism; range panics.
// ---------------------------------------------------------------------------

// tabularDense is an environment with a non-one-hot (dense) observation.
type tabularDense struct{}

func (tabularDense) NumActions() int  { return 2 }
func (tabularDense) ObsDim() int      { return 3 }
func (tabularDense) Reset() []float64 { return []float64{0.5, 0.5, 0.5} }
func (tabularDense) Step(a int) ([]float64, float64, bool) {
	return []float64{0.5, 0.5, 0.5}, 1, true
}

func TestTabularNonOneHotErrors(t *testing.T) {
	env := tabularDense{}
	agent := rl.NewQLearning(env, 0.5, 0.9, 0.1, 1)
	_, err := agent.Episode(env)
	if err == nil {
		t.Fatal("expected clear error for non-one-hot observation without WithStateIndexFunc")
	}
	t.Logf("non-one-hot error: %v", err)

	// with a custom index function the same env works.
	agent2 := rl.NewQLearning(env, 0.5, 0.9, 0.1, 1,
		rl.WithNumStates(1),
		rl.WithStateIndexFunc(func(obs []float64) (int, error) { return 0, nil }))
	if _, err := agent2.Episode(env); err != nil {
		t.Fatalf("custom state-index func should work: %v", err)
	}
}

func TestTabularDeterminism(t *testing.T) {
	run := func() [][]float64 {
		env := rl.NewChain(5, 20)
		agent := rl.NewSARSA(env, 0.4, 0.9, 0.2, 77)
		for ep := 0; ep < 300; ep++ {
			if _, err := agent.Episode(env); err != nil {
				t.Fatal(err)
			}
		}
		return agent.QTable()
	}
	a, b := run(), run()
	for s := range a {
		for act := range a[s] {
			if a[s][act] != b[s][act] {
				t.Fatalf("non-deterministic: Q[%d][%d] %v vs %v", s, act, a[s][act], b[s][act])
			}
		}
	}
}

func TestTabularRangePanics(t *testing.T) {
	env := rl.NewChain(5, 20)
	cases := []struct {
		name                  string
		alpha, gamma, epsilon float64
	}{
		{"alpha>1", 1.5, 0.9, 0.1},
		{"alpha<0", -0.1, 0.9, 0.1},
		{"gamma>1", 0.5, 1.5, 0.1},
		{"epsilon<0", 0.5, 0.9, -0.1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("expected panic for %s", c.name)
				}
			}()
			rl.NewQLearning(env, c.alpha, c.gamma, c.epsilon, 1)
		})
	}
}
