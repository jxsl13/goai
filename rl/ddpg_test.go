package rl

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// ddpgRandBatch builds a fixed, deterministic (states, actions, y) batch for the
// critic gradcheck.
func ddpgRandBatch(rng *rand.Rand, n, obsDim, actDim int) (states, actions, y *tensor.Tensor) {
	s := make([][]float64, n)
	a := make([][]float64, n)
	for i := 0; i < n; i++ {
		s[i] = make([]float64, obsDim)
		for j := range s[i] {
			s[i][j] = rng.NormFloat64()
		}
		a[i] = make([]float64, actDim)
		for j := range a[i] {
			a[i][j] = 2*rng.Float64() - 1
		}
	}
	yt := tensor.New(tensor.F64, tensor.Shape{n, 1})
	for i := 0; i < n; i++ {
		yt.SetF64(rng.NormFloat64(), i, 0)
	}
	return contMat(s), contMat(a), yt
}

// TestDDPGActWithinBounds checks the deterministic evaluation action is the right
// width and always inside the action box, for many observations.
func TestDDPGActWithinBounds(t *testing.T) {
	env := NewPointMass(2, 3)
	agent := NewDDPG(env, WithDDPGHidden(16), WithDDPGSeed(5))
	low, high := env.ActionBounds()
	obs := env.Reset()
	for step := 0; step < 200; step++ {
		a, err := agent.Act(obs)
		if err != nil {
			t.Fatal(err)
		}
		if len(a) != env.ActionDim() {
			t.Fatalf("action dim %d, want %d", len(a), env.ActionDim())
		}
		for i := range a {
			if a[i] < low[i]-1e-9 || a[i] > high[i]+1e-9 {
				t.Fatalf("action[%d]=%v out of [%v,%v]", i, a[i], low[i], high[i])
			}
		}
		obs, _, _ = env.Step(a)
	}
}

// TestDDPGCriticGradcheck finite-difference-checks the differentiated critic
// (Bellman MSE) loss w.r.t. every critic parameter on a fixed batch (§V16 gradcheck
// anchor): the autograd gradient must match central differences to ~1e-4. The
// critic is temporarily swapped for a tanh-activated MLP of the same shape so the
// central differences do not straddle ReLU kinks (the loss COMPOSITION under test —
// concat → MLP → MSE and its VJPs — is identical; only the smooth activation makes
// the finite-difference reference well-defined).
func TestDDPGCriticGradcheck(t *testing.T) {
	env := NewPointMass(1, 1) // obsDim 2, actDim 1 → tiny critic
	agent := NewDDPG(env, WithDDPGHidden(4), WithDDPGSeed(9))
	agent.Critic = nn.NewSequential(
		nn.NewLinear(tensor.F64, env.ObsDim()+env.ActionDim(), 4, 1),
		nn.Tanh(),
		nn.NewLinear(tensor.F64, 4, 4, 2),
		nn.Tanh(),
		nn.NewLinear(tensor.F64, 4, 1, 3),
	)
	rng := rand.New(rand.NewPCG(7, 7))
	states, actions, y := ddpgRandBatch(rng, 5, env.ObsDim(), env.ActionDim())

	lossVal := func() float64 {
		l, err := agent.ddpgCriticLoss(backend.NewContext(), states, actions, y)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}

	// Analytic gradients via the tape.
	tape := autograd.NewTape()
	l, err := agent.ddpgCriticLoss(tape.Context(), states, actions, y)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}

	const eps = 1e-6
	var maxRel float64
	var checked int
	for _, p := range agent.Critic.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatal("nil gradient for a critic parameter")
		}
		for j := 0; j < p.Numel(); j++ {
			idx := tensor.Unravel(j, p.Shape())
			orig := p.AtF64(idx...)
			p.SetF64(orig+eps, idx...)
			lp := lossVal()
			p.SetF64(orig-eps, idx...)
			lm := lossVal()
			p.SetF64(orig, idx...)
			num := (lp - lm) / (2 * eps)
			ana := g.AtF64(idx...)
			rel := math.Abs(ana-num) / (math.Abs(num) + 1e-9)
			if rel > maxRel {
				maxRel = rel
			}
			checked++
		}
	}
	t.Logf("DDPG critic gradcheck: %d params, max rel err %g", checked, maxRel)
	if maxRel > 1e-4 {
		t.Errorf("critic gradcheck max rel err %g > 1e-4", maxRel)
	}
}

// TestDDPGConverges is the primary anchor: a trained DDPG agent must solve the
// point-mass task (drive to the origin) and beat a random policy by a wide margin,
// with the learning curve rising over training.
func TestDDPGConverges(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping convergence test in -short mode")
	}
	const seed = 3
	env := NewPointMass(1, seed)
	agent := NewDDPG(env,
		WithDDPGHidden(64), WithDDPGSeed(seed),
		WithDDPGActorLR(1e-3), WithDDPGCriticLR(1e-3),
		WithDDPGBatchSize(64), WithDDPGBufferSize(20000),
		WithDDPGNoise(0.2),
	)

	// Random-policy baseline (uniform in the action box).
	randEnv := NewPointMass(1, 777)
	randRng := rand.New(rand.NewPCG(777, 1))
	low, high := randEnv.ActionBounds()
	var randRet float64
	const evalEpisodes = 20
	for e := 0; e < evalEpisodes; e++ {
		randRet += evalPolicy(randEnv, func(obs []float64) []float64 {
			a := make([]float64, len(low))
			for i := range a {
				a[i] = low[i] + (high[i]-low[i])*randRng.Float64()
			}
			return a
		})
	}
	randRet /= evalEpisodes

	// Train; record early vs late learning-curve returns.
	var early, late float64
	const episodes = 120
	for ep := 0; ep < episodes; ep++ {
		ret, err := agent.Episode(env)
		if err != nil {
			t.Fatal(err)
		}
		if ep < 10 {
			early += ret / 10
		}
		if ep >= episodes-10 {
			late += ret / 10
		}
	}

	// Deterministic evaluation of the trained policy on held-out start states.
	evalEnv := NewPointMass(1, 555)
	var trainedRet float64
	for e := 0; e < evalEpisodes; e++ {
		trainedRet += evalPolicy(evalEnv, func(obs []float64) []float64 {
			a, err := agent.Act(obs)
			if err != nil {
				t.Fatal(err)
			}
			return a
		})
	}
	trainedRet /= evalEpisodes

	t.Logf("DDPG returns — random %.3f, early %.3f, late %.3f, trained-eval %.3f",
		randRet, early, late, trainedRet)

	if late <= early {
		t.Errorf("learning curve did not rise: early %.3f, late %.3f", early, late)
	}
	// Trained policy must clearly beat random (returns are negative costs → closer
	// to 0 is better). Require closing most of the gap toward the ~0 optimum.
	if trainedRet <= randRet+0.5*math.Abs(randRet) {
		t.Errorf("trained return %.3f does not clearly beat random %.3f", trainedRet, randRet)
	}
}
