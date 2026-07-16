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

// sacLogProbRef is the independent squashed-Gaussian log-probability reference:
// diagonal Gaussian density of u, minus the tanh Jacobian Σ log(1−tanh²u), minus
// the affine-rescale Jacobian Σ log(scale).
func sacLogProbRef(mean, logStd, u, scale []float64) float64 {
	var lp float64
	for i := range mean {
		std := math.Exp(logStd[i])
		z := (u[i] - mean[i]) / std
		gauss := -0.5*z*z - logStd[i] - 0.5*math.Log(2*math.Pi)
		th := math.Tanh(u[i])
		lp += gauss - math.Log(1-th*th) - math.Log(scale[i])
	}
	return lp
}

// TestSACLogProbTanhCorrection hand-checks the squashed-Gaussian log-prob at 1e-8,
// including the tanh change-of-variables term log(1−tanh²u) and the affine-scale
// Jacobian (§V16 tanh-logprob anchor). Two cases: unit scale (bounds [−1,1], the
// tanh term isolated) and a 2-D non-unit scale (bounds [−2,2]).
func TestSACLogProbTanhCorrection(t *testing.T) {
	// Case 1: 1-D, bounds [−1,1] → scale=1, log(scale)=0.
	s1 := NewSAC(NewPointMass(1, 1), WithSACSeed(1))
	mean, logStd, u := []float64{0.3}, []float64{-0.5}, []float64{0.7}
	got1, err := s1.sacLogProb(backend.NewContext(),
		contMat([][]float64{mean}), contMat([][]float64{logStd}), contMat([][]float64{u}))
	if err != nil {
		t.Fatal(err)
	}
	want1 := sacLogProbRef(mean, logStd, u, []float64{1})
	if math.Abs(got1.AtF64(0, 0)-want1) > 1e-8 {
		t.Errorf("1-D log-prob = %.12g, want %.12g", got1.AtF64(0, 0), want1)
	}

	// Case 2: 2-D, bounds [−2,2] → scale=2 per dim (exercises the Σlog(scale) offset
	// and the multi-dimension sum).
	env2 := NewPointMass(2, 1)
	env2.ForceMax = 2
	s2 := NewSAC(env2, WithSACSeed(1))
	mean2, logStd2, u2 := []float64{0.1, -0.4}, []float64{0.2, -1.0}, []float64{-0.6, 0.9}
	got2, err := s2.sacLogProb(backend.NewContext(),
		contMat([][]float64{mean2}), contMat([][]float64{logStd2}), contMat([][]float64{u2}))
	if err != nil {
		t.Fatal(err)
	}
	want2 := sacLogProbRef(mean2, logStd2, u2, []float64{2, 2})
	if math.Abs(got2.AtF64(0, 0)-want2) > 1e-8 {
		t.Errorf("2-D log-prob = %.12g, want %.12g", got2.AtF64(0, 0), want2)
	}
}

// TestSACActionsWithinBounds checks that both the deterministic Act and the
// stochastic rollout sampler always emit actions inside the box, over many draws.
func TestSACActionsWithinBounds(t *testing.T) {
	env := NewPointMass(2, 4)
	env.ForceMax = 2
	s := NewSAC(env, WithSACHidden(16), WithSACSeed(6))
	low, high := env.ActionBounds()
	inBounds := func(a []float64) {
		t.Helper()
		if len(a) != env.ActionDim() {
			t.Fatalf("dim %d, want %d", len(a), env.ActionDim())
		}
		for i := range a {
			if a[i] < low[i]-1e-9 || a[i] > high[i]+1e-9 {
				t.Fatalf("action[%d]=%v out of [%v,%v]", i, a[i], low[i], high[i])
			}
		}
	}
	obs := env.Reset()
	for i := 0; i < 300; i++ {
		det, err := s.Act(obs)
		if err != nil {
			t.Fatal(err)
		}
		inBounds(det)
		stoch, err := s.rolloutAction(obs)
		if err != nil {
			t.Fatal(err)
		}
		inBounds(stoch)
	}
}

// TestSACClippedDoubleQ verifies the soft Bellman target uses min(Q1',Q2') (§V16
// clipped-double-Q anchor). Driving one target critic's output to +∞ leaves the
// target BOUNDED (min ignores the inflated critic — the whole point, vs. max/mean),
// while driving it to −∞ makes the target explode negative (min selects it). The
// sampling RNG is reset before each call so the sampled next-actions match.
func TestSACClippedDoubleQ(t *testing.T) {
	env := NewPointMass(1, 1) // obsDim 2
	s := NewSAC(env, WithSACHidden(8), WithSACSeed(2))
	next := contMat([][]float64{{0.1, 0.2}, {-0.3, 0.4}})
	rewards := []float64{0.5, -0.2}
	dones := []bool{false, false}
	resetRng := func() { s.rng = rand.New(rand.NewPCG(2, 2)) }

	setLastBias := func(net *nn.Sequential, v float64) {
		ps := net.Params()
		ps[len(ps)-1].SetF64(v, 0) // final Linear bias [1]
	}

	// Q2' → −1e6: min picks it, targets explode negative.
	setLastBias(s.Target2, -1e6)
	resetRng()
	yNeg, err := s.sacCriticTargets(next, rewards, dones)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(rewards); i++ {
		if yNeg.AtF64(i, 0) > -1e5 {
			t.Errorf("target[%d]=%v not driven negative by min(Q1',−1e6)", i, yNeg.AtF64(i, 0))
		}
	}

	// Q2' → +1e6: min ignores it (picks the bounded Q1'), targets stay bounded.
	setLastBias(s.Target2, 1e6)
	resetRng()
	yPos, err := s.sacCriticTargets(next, rewards, dones)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(rewards); i++ {
		if math.Abs(yPos.AtF64(i, 0)) > 100 {
			t.Errorf("target[%d]=%v not bounded — min failed to clip the inflated critic", i, yPos.AtF64(i, 0))
		}
	}
}

// TestSACCriticGradcheck finite-difference-checks the twin-critic Bellman loss
// w.r.t. both critics' parameters on a fixed batch (§V16 gradcheck anchor). The
// critics are swapped for tanh MLPs of the same shape so the finite differences do
// not straddle ReLU kinks (identical loss composition: concat → MLP → MSE, ×2).
func TestSACCriticGradcheck(t *testing.T) {
	env := NewPointMass(1, 1)
	s := NewSAC(env, WithSACHidden(4), WithSACSeed(3))
	cin := env.ObsDim() + env.ActionDim()
	s.Critic1 = nn.NewSequential(
		nn.NewLinear(tensor.F64, cin, 4, 1), nn.Tanh(),
		nn.NewLinear(tensor.F64, 4, 4, 2), nn.Tanh(),
		nn.NewLinear(tensor.F64, 4, 1, 3),
	)
	s.Critic2 = nn.NewSequential(
		nn.NewLinear(tensor.F64, cin, 4, 11), nn.Tanh(),
		nn.NewLinear(tensor.F64, 4, 4, 12), nn.Tanh(),
		nn.NewLinear(tensor.F64, 4, 1, 13),
	)
	rng := rand.New(rand.NewPCG(5, 5))
	states, actions, y := ddpgRandBatch(rng, 5, env.ObsDim(), env.ActionDim())

	lossVal := func() float64 {
		l, err := s.sacCriticLoss(backend.NewContext(), states, actions, y)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	tape := autograd.NewTape()
	l, err := s.sacCriticLoss(tape.Context(), states, actions, y)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}

	const eps = 1e-6
	var maxRel float64
	var checked int
	for _, net := range []*nn.Sequential{s.Critic1, s.Critic2} {
		for _, p := range net.Params() {
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
				rel := math.Abs(g.AtF64(idx...)-num) / (math.Abs(num) + 1e-9)
				if rel > maxRel {
					maxRel = rel
				}
				checked++
			}
		}
	}
	t.Logf("SAC twin-critic gradcheck: %d params, max rel err %g", checked, maxRel)
	if maxRel > 1e-4 {
		t.Errorf("critic gradcheck max rel err %g > 1e-4", maxRel)
	}
}

// TestSACConverges is the primary anchor for SAC: a trained agent solves the
// point-mass task and beats a random policy by a wide margin, with the learning
// curve rising over training.
func TestSACConverges(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping convergence test in -short mode")
	}
	const seed = 4
	env := NewPointMass(1, seed)
	s := NewSAC(env,
		WithSACHidden(64), WithSACSeed(seed), WithSACLR(1e-3),
		WithSACBatchSize(64), WithSACBufferSize(20000), WithSACAlpha(0.1),
	)

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

	var early, late float64
	const episodes = 120
	for ep := 0; ep < episodes; ep++ {
		ret, err := s.Episode(env)
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

	evalEnv := NewPointMass(1, 555)
	var trainedRet float64
	for e := 0; e < evalEpisodes; e++ {
		trainedRet += evalPolicy(evalEnv, func(obs []float64) []float64 {
			a, err := s.Act(obs)
			if err != nil {
				t.Fatal(err)
			}
			return a
		})
	}
	trainedRet /= evalEpisodes

	t.Logf("SAC returns — random %.3f, early %.3f, late %.3f, trained-eval %.3f",
		randRet, early, late, trainedRet)
	if late <= early {
		t.Errorf("learning curve did not rise: early %.3f, late %.3f", early, late)
	}
	if trainedRet <= randRet+0.5*math.Abs(randRet) {
		t.Errorf("trained return %.3f does not clearly beat random %.3f", trainedRet, randRet)
	}
}

// TestSACAutoAlpha checks that automatic entropy-temperature tuning actually moves
// α away from its initial value over training (the auto-α controller is live).
func TestSACAutoAlpha(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping training test in -short mode")
	}
	env := NewPointMass(1, 8)
	s := NewSAC(env,
		WithSACHidden(32), WithSACSeed(8), WithSACLR(1e-3),
		WithSACBatchSize(64), WithSACBufferSize(10000),
		WithSACAlpha(0.2), WithSACAutoAlpha(true),
	)
	if s.logAlpha == nil {
		t.Fatal("auto-alpha enabled but logAlpha not allocated")
	}
	start := s.Alpha
	for ep := 0; ep < 40; ep++ {
		if _, err := s.Episode(env); err != nil {
			t.Fatal(err)
		}
	}
	if math.Abs(s.Alpha-start) < 1e-4 {
		t.Errorf("auto-alpha did not move: start %v, end %v", start, s.Alpha)
	}
	t.Logf("auto-alpha: start %.4f → end %.4f", start, s.Alpha)
}
