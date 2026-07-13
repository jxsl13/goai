package rl_test

import (
	"fmt"

	"github.com/jxsl13/goai/rl"
)

// --- Level 1: trivial ---------------------------------------------------------

// DiscountedReturns computes the reward-to-go G_t = r_t + γ·G_{t+1} for each step.
// With three unit rewards and γ=0.5 the last step is worth 1, and earlier steps
// add a half-weighted tail: 1.5 and 1.75.
func ExampleDiscountedReturns() {
	g := rl.DiscountedReturns([]float64{1, 1, 1}, 0.5)
	fmt.Println(g)
	// Output: [1.75 1.5 1]
}

// --- Level 2: realistic use case ----------------------------------------------

// The Chain environment is a line of states; the agent starts in the middle and
// the right end pays 1.0. From the center of a 5-state chain two steps right reach
// the goal: the first pays nothing and continues, the second pays 1 and ends the
// episode with a one-hot observation on the rightmost state.
func ExampleChain() {
	env := rl.NewChain(5 /*states*/, 20 /*maxSteps*/)
	env.Reset() // start in the middle (state 2)

	_, r1, done1 := env.Step(1)   // move right → state 3
	obs, r2, done2 := env.Step(1) // move right → state 4 (right end)
	fmt.Println(r1, done1, "|", r2, done2, obs)
	// Output: 0 false | 1 true [0 0 0 0 1]
}

// --- Level 3: embedded — train an agent to solve the task ---------------------

// A full learning loop: a DQN plays 400 episodes on the chain, then acts greedily.
// From the start state its learned best move is "right" (action 1) — the optimal
// policy — showing value-based control end to end (deterministic via the seed).
func Example_dqnLearnsChain() {
	env := rl.NewChain(5, 20)
	agent := rl.NewDQN(env, 16 /*hidden*/, 0.01 /*lr*/, 0.9 /*γ*/, 7 /*seed*/)
	for range 400 {
		if _, err := agent.Episode(env); err != nil {
			panic(err)
		}
	}

	start := []float64{0, 0, 1, 0, 0} // one-hot at the middle state
	a, err := agent.GreedyAction(start, 2)
	if err != nil {
		panic(err)
	}
	fmt.Println("greedy action:", a) // 1 = right
	// Output: greedy action: 1
}

// GAE turns a trajectory of rewards and value estimates into the advantages a
// policy-gradient method (e.g. PPO via nn.PPOClipLoss) trains on. With γ=1 and
// λ=1 the advantage is the plain Monte-Carlo return minus the baseline value;
// here the baselines are 0, so each advantage is just the remaining reward-to-go.
func ExampleGAE() {
	rewards := []float64{1, 1, 1}
	values := []float64{0, 0, 0}
	dones := []bool{false, false, false}

	adv, returns := rl.GAE(rewards, values, dones, 0 /*bootstrap*/, false, 1.0 /*γ*/, 1.0 /*λ*/)
	fmt.Println("advantages:", adv)
	fmt.Println("returns:   ", returns)
	// Output:
	// advantages: [3 2 1]
	// returns:    [3 2 1]
}

// An environment exposes its geometry (observation width, action count) so
// agents can size their networks; DQN.QValues inspects the learned per-action
// values for one observation — the greedy policy is its argmax.
func ExampleDQN_QValues() {
	env := rl.NewChain(4, 20)
	fmt.Println(env.ObsDim(), env.NumActions())
	agent := rl.NewDQN(env, 8, 0.01, 0.9, 1)
	q, _ := agent.QValues(env.Reset(), env.NumActions())
	fmt.Println(len(q))
	// Output:
	// 4 2
	// 2
}
