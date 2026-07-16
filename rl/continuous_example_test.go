package rl_test

import (
	"fmt"

	"github.com/jxsl13/goai/rl"
)

// The PointMass environment exposes its continuous geometry — observation width,
// action width and the per-dimension action box — and steps under a real-valued
// action. From rest at the origin a zero action stays put and earns zero cost.
func ExamplePointMass() {
	env := rl.NewPointMass(2 /*dims*/, 1 /*seed*/)
	fmt.Println("obs", env.ObsDim(), "act", env.ActionDim())
	low, high := env.ActionBounds()
	fmt.Println("bounds", low[0], high[0])

	env.Reset()
	_, _, done := env.Step([]float64{0, 0})
	fmt.Println("done", done)
	// Output:
	// obs 4 act 2
	// bounds -1 1
	// done false
}

// ExampleDDPG constructs a DDPG agent for the point-mass task, runs one interaction
// episode (the learning loop; the return climbs toward the ~0 optimum with more
// training) and reads back the deterministic evaluation action — always inside the
// action box.
func ExampleDDPG() {
	env := rl.NewPointMass(1, 1)
	agent := rl.NewDDPG(env, rl.WithDDPGHidden(16), rl.WithDDPGSeed(1))
	if _, err := agent.Episode(env); err != nil {
		panic(err)
	}
	a, err := agent.Act(env.Reset())
	if err != nil {
		panic(err)
	}
	low, high := env.ActionBounds()
	fmt.Println(len(a) == env.ActionDim() && a[0] >= low[0] && a[0] <= high[0])
	// Output: true
}

// ExampleSAC constructs a Soft Actor-Critic agent, runs one interaction episode and
// reads back the deterministic (mean) evaluation action, which always lies within
// the environment's action bounds.
func ExampleSAC() {
	env := rl.NewPointMass(1, 1)
	agent := rl.NewSAC(env, rl.WithSACHidden(16), rl.WithSACSeed(1))
	if _, err := agent.Episode(env); err != nil {
		panic(err)
	}
	a, err := agent.Act(env.Reset())
	if err != nil {
		panic(err)
	}
	low, high := env.ActionBounds()
	fmt.Println(len(a) == env.ActionDim() && a[0] >= low[0] && a[0] <= high[0])
	// Output: true
}
