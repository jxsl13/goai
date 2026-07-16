package rl

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// ppoFixedBatch builds a small deterministic rollout at the agent's current
// (untrained) parameters and returns the pieces the loss builders consume, with
// logpOld equal to the current policy log-probs (so the PPO ratio ρ = 1 exactly at
// the unperturbed point — what holds on the first epoch of every real Update).
func ppoFixedBatch(t *testing.T, actor, critic *nn.Sequential, seed uint64) (states [][]float64, actions []int, oldT, advT *tensor.Tensor) {
	t.Helper()
	env := NewChain(5, 20)
	rng := rand.New(rand.NewPCG(seed, 0x9b0))
	ro, err := rlRollout(env, actor, critic, rng, 24)
	if err != nil {
		t.Fatal(err)
	}
	adv, _ := GAE(ro.rewards, ro.values, ro.dones, 0, true, 0.99, 0.95)
	// logpOld = current log πθ(a|s): recompute (no tape) so ρ=1 at these params.
	logits, err := forward(backend.NewContext(), actor, ro.states)
	if err != nil {
		t.Fatal(err)
	}
	lp, err := nn.TokenLogProbs(backend.NewContext(), logits, ro.actions)
	if err != nil {
		t.Fatal(err)
	}
	old := make([]float64, len(ro.actions))
	for i := range old {
		old[i] = lp.AtF64(i)
	}
	return ro.states, ro.actions, rlVec(old), rlNormAdv(adv)
}

// §V16 anchor 1 — CONVERGES TO OPTIMAL: a trained PPO agent's average episode
// return on Chain reaches the near-optimal level (≥0.9 of the achievable 1.0) and
// clearly beats its own untrained/random performance.
func TestPPOAgentLearnsChain(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a PPO agent; skipped in -short")
	}
	env := NewChain(5, 20)
	agent := NewPPO(env, 16, 0.02, 0.99, 7, WithPPORolloutSteps(200))

	const iters = 60
	var early, late float64
	for it := range iters {
		r, err := agent.Update(env)
		if err != nil {
			t.Fatal(err)
		}
		if it < 5 {
			early += r / 5
		}
		if it >= iters-5 {
			late += r / 5
		}
	}
	t.Logf("PPO learning curve: early avg return %.3f -> late %.3f", early, late)
	if late < 0.9 {
		t.Fatalf("PPO did not converge to near-optimal: late avg %.3f", late)
	}
	if late <= early+0.05 {
		t.Fatalf("PPO did not clearly beat its untrained self: %.3f -> %.3f", early, late)
	}
	// converged policy is greedy-right from the interior + critic ≈ discounted optimal.
	probs, err := agent.ActionProbs([]float64{0, 0, 1, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if probs[1] <= probs[0] {
		t.Fatalf("policy not right-preferring at start: P=%v", probs)
	}
	v0, err := agent.Value([]float64{0, 0, 1, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	optimal := math.Pow(0.99, 2) // 2 steps right to terminal reward 1
	t.Logf("critic V(start) = %.3f (optimal ≈ %.3f)", v0, optimal)
	if math.Abs(v0-optimal) > 0.4 {
		t.Fatalf("critic value %.3f far from optimal %.3f", v0, optimal)
	}
}

// §V16 anchor 4 — VALUE BASELINE / A2C also learns: the A2C baseline agent also
// converges to near-optimal on Chain (the value baseline is what lets a single
// low-variance gradient step per rollout succeed).
func TestA2CAgentLearnsChain(t *testing.T) {
	if testing.Short() {
		t.Skip("trains an A2C agent; skipped in -short")
	}
	env := NewChain(5, 20)
	agent := NewA2C(env, 16, 0.02, 0.99, 7, WithA2CRolloutSteps(200))

	const iters = 120
	var early, late float64
	for it := range iters {
		r, err := agent.Update(env)
		if err != nil {
			t.Fatal(err)
		}
		if it < 5 {
			early += r / 5
		}
		if it >= iters-5 {
			late += r / 5
		}
	}
	t.Logf("A2C learning curve: early avg return %.3f -> late %.3f", early, late)
	if late < 0.9 {
		t.Fatalf("A2C did not converge to near-optimal: late avg %.3f", late)
	}
	if late <= early+0.05 {
		t.Fatalf("A2C did not clearly beat its untrained self: %.3f -> %.3f", early, late)
	}
}

// §V16 anchor 2 — GAE hand-check (self-contained): the advantage matches the
// exponentially-weighted TD-residual sum Â_t = Σ_l (γλ)^l δ_{t+l} on a tiny known
// trajectory at 1e-10, and returns = advantages + values.
func TestPPOGAEHandComputed(t *testing.T) {
	gamma, lam := 0.9, 0.8
	rewards := []float64{1.0, 0.0, 2.0}
	values := []float64{0.5, 0.7, 0.3}
	dones := []bool{false, false, true} // terminal after the last step → no bootstrap
	adv, ret := GAE(rewards, values, dones, 0, true, gamma, lam)

	// deltas (last step terminal → nextV=0, nonterminal=0):
	// δ0 = 1 + 0.9·0.7 − 0.5 = 1.13
	// δ1 = 0 + 0.9·0.3 − 0.7 = -0.43
	// δ2 = 2 + 0        − 0.3 = 1.7
	d0 := 1.0 + gamma*0.7 - 0.5
	d1 := 0.0 + gamma*0.3 - 0.7
	d2 := 2.0 - 0.3
	// Â2 = δ2; Â1 = δ1 + γλ·δ2; Â0 = δ0 + γλ·δ1 + (γλ)²·δ2
	gl := gamma * lam
	want := []float64{d0 + gl*d1 + gl*gl*d2, d1 + gl*d2, d2}
	for i := range want {
		if math.Abs(adv[i]-want[i]) > 1e-10 {
			t.Errorf("adv[%d] = %.12g, want %.12g", i, adv[i], want[i])
		}
		if math.Abs(ret[i]-(adv[i]+values[i])) > 1e-12 {
			t.Errorf("return[%d] != adv+value", i)
		}
	}
}

// §V16 anchor 2 (limits) — λ=0 collapses GAE to the one-step TD residual δ_t, and
// λ=1 collapses it to the Monte-Carlo advantage (discounted return − V).
func TestPPOGAELambdaLimits(t *testing.T) {
	gamma := 0.9
	rewards := []float64{1.0, 0.0, 2.0, -1.0}
	values := []float64{0.5, 0.7, 0.3, 0.2}
	next := 0.4
	dones := make([]bool, 4)

	adv0, _ := GAE(rewards, values, dones, next, false, gamma, 0)
	for i := range rewards {
		nv := next
		if i+1 < len(rewards) {
			nv = values[i+1]
		}
		td := rewards[i] + gamma*nv - values[i]
		if math.Abs(adv0[i]-td) > 1e-12 {
			t.Errorf("λ=0 adv[%d]=%.12g want TD δ=%.12g", i, adv0[i], td)
		}
	}

	adv1, _ := GAE(rewards, values, dones, next, false, gamma, 1)
	n := len(rewards)
	for i := range rewards {
		g := 0.0
		for k := i; k < n; k++ {
			g += math.Pow(gamma, float64(k-i)) * rewards[k]
		}
		g += math.Pow(gamma, float64(n-i)) * next
		want := g - values[i]
		if math.Abs(adv1[i]-want) > 1e-10*math.Max(1, math.Abs(want)) {
			t.Errorf("λ=1 adv[%d]=%.12g want MC(G−V)=%.12g", i, adv1[i], want)
		}
	}
}

// §V16 anchor 3 — CLIP mechanism: for a positive advantage with ρ pushed above the
// trust region, the clipped surrogate objective is strictly below the unclipped
// one (clipping bounds the update). Realized via nn.PPOClipLoss with ε=0.2 vs a
// huge ε (no clipping).
func TestPPOClipBoundsObjective(t *testing.T) {
	// one sample: ρ = exp(logNew−logOld) = 1.5, Â = +2, ε = 0.2.
	logNew := rlVec([]float64{math.Log(1.5)})
	logOld := rlVec([]float64{0})
	adv := rlVec([]float64{2.0})

	clip, err := nn.PPOClipLoss(backend.NewContext(), logNew, logOld, adv, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	unclip, err := nn.PPOClipLoss(backend.NewContext(), logNew, logOld, adv, 1e9)
	if err != nil {
		t.Fatal(err)
	}
	// objective = −loss. clipped objective ≤ unclipped objective.
	clipObj, uncObj := -clip.AtF64(), -unclip.AtF64()
	wantClip := math.Min(1.5, 1.2) * 2.0 // = 2.4
	wantUnc := 1.5 * 2.0                 // = 3.0
	if math.Abs(clipObj-wantClip) > 1e-12 || math.Abs(uncObj-wantUnc) > 1e-12 {
		t.Fatalf("objectives clip=%.4f (want %.4f) unclip=%.4f (want %.4f)", clipObj, wantClip, uncObj, wantUnc)
	}
	if !(clipObj < uncObj) {
		t.Fatalf("clip did not bound the objective: clip %.4f ≮ unclip %.4f", clipObj, uncObj)
	}
}

// §V16 anchor 3 (collapse) — PPO with clipping disabled (huge ε) and one epoch and
// no entropy produces the SAME policy gradient as the A2C/vanilla-PG objective on
// the same batch (at collection-time params ρ=1, so PPO's first-epoch gradient
// −Â·ρ/N coincides with A2C's −Â/N exactly).
func TestPPOCollapsesToA2C(t *testing.T) {
	actor := policyNet(5, 8, 2, 3)
	critic := nn.NewSequential(nn.NewLinear(tensor.F64, 5, 8, 5), nn.ReLU(), nn.NewLinear(tensor.F64, 8, 1, 6))
	states, actions, oldT, advT := ppoFixedBatch(t, actor, critic, 11)

	gPPO := gradOf(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return ppoPolicyLoss(ctx, actor, states, actions, oldT, advT, 1e9 /*no clip*/, 0 /*no entropy*/)
	}, actor.Params())
	gA2C := gradOf(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return a2cPolicyLoss(ctx, actor, states, actions, advT, 0)
	}, actor.Params())

	var maxErr float64
	for p := range gPPO {
		for i := range gPPO[p] {
			maxErr = math.Max(maxErr, math.Abs(gPPO[p][i]-gA2C[p][i]))
		}
	}
	t.Logf("PPO(no-clip,1-epoch) vs A2C gradient max |Δ| = %.2e", maxErr)
	if maxErr > 1e-10 {
		t.Fatalf("PPO did not collapse to A2C: gradient max |Δ| = %.2e", maxErr)
	}
}

// §V16 anchor 5 — GRADCHECK: the PPO policy loss (clipped surrogate + entropy) is
// differentiable through the nn tape; the analytic gradient w.r.t. every actor
// parameter matches central finite differences to ~1e-4 on a fixed batch.
func TestPPOPolicyGradcheck(t *testing.T) {
	actor := policyNet(5, 8, 2, 4)
	critic := nn.NewSequential(nn.NewLinear(tensor.F64, 5, 8, 7), nn.ReLU(), nn.NewLinear(tensor.F64, 8, 1, 8))
	states, actions, oldT, advT := ppoFixedBatch(t, actor, critic, 13)

	lossFn := func(ctx *backend.Context) (*tensor.Tensor, error) {
		return ppoPolicyLoss(ctx, actor, states, actions, oldT, advT, 0.2, 0.01)
	}
	analytic := gradOf(t, lossFn, actor.Params())

	const h = 1e-5
	var maxRel float64
	for pi, p := range actor.Params() {
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			orig := p.AtF64(idx...)
			p.SetF64(orig+h, idx...)
			lp, err := lossFn(backend.NewContext())
			if err != nil {
				t.Fatal(err)
			}
			p.SetF64(orig-h, idx...)
			lm, err := lossFn(backend.NewContext())
			if err != nil {
				t.Fatal(err)
			}
			p.SetF64(orig, idx...)
			fd := (lp.AtF64() - lm.AtF64()) / (2 * h)
			rel := math.Abs(fd-analytic[pi][i]) / (1 + math.Abs(fd))
			maxRel = math.Max(maxRel, rel)
		}
	}
	t.Logf("PPO policy-loss gradcheck max rel err = %.2e", maxRel)
	if maxRel > 1e-4 {
		t.Fatalf("gradcheck failed: max rel err %.2e", maxRel)
	}
}

// §V16 anchor 6 — determinism: identical seeds reproduce the whole learning curve
// (and thus every trajectory and update) bit-for-bit.
func TestPPODeterministic(t *testing.T) {
	run := func() []float64 {
		env := NewChain(5, 20)
		agent := NewPPO(env, 8, 0.02, 0.99, 21, WithPPORolloutSteps(64))
		out := make([]float64, 8)
		for i := range out {
			r, err := agent.Update(env)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = r
		}
		return out
	}
	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("nondeterministic: update %d %v != %v", i, a[i], b[i])
		}
	}
}

// Sanity — constructor defaults, option ranges, and observed dims.
func TestPPOSanity(t *testing.T) {
	env := NewChain(5, 20)
	p := NewPPO(env, 16, 0.02, 0.99, 1)
	if p.Clip != 0.2 || p.Epochs != 4 || p.Lambda != 0.95 || p.ValueCoef != 0.5 || p.EntropyCoef != 0.01 || p.RolloutSteps != 256 {
		t.Fatalf("unexpected defaults: %+v", p)
	}
	// invalid option values are ignored (defaults preserved).
	p2 := NewPPO(env, 16, 0.02, 0.99, 1,
		WithPPOClip(-1), WithPPOEpochs(0), WithPPOLambda(2), WithPPOValueCoef(-1), WithPPOEntropyCoef(-1), WithPPORolloutSteps(0))
	if p2.Clip != 0.2 || p2.Epochs != 4 || p2.Lambda != 0.95 || p2.ValueCoef != 0.5 || p2.EntropyCoef != 0.01 || p2.RolloutSteps != 256 {
		t.Fatalf("invalid options changed defaults: %+v", p2)
	}
	// valid overrides apply.
	p3 := NewPPO(env, 16, 0.02, 0.99, 1,
		WithPPOClip(0.3), WithPPOEpochs(8), WithPPOLambda(0.9), WithPPOValueCoef(1), WithPPOEntropyCoef(0), WithPPORolloutSteps(128))
	if p3.Clip != 0.3 || p3.Epochs != 8 || p3.Lambda != 0.9 || p3.ValueCoef != 1 || p3.EntropyCoef != 0 || p3.RolloutSteps != 128 {
		t.Fatalf("valid options not applied: %+v", p3)
	}
	probs, err := p.ActionProbs(env.Reset())
	if err != nil {
		t.Fatal(err)
	}
	if len(probs) != env.NumActions() {
		t.Fatalf("ActionProbs len %d != NumActions %d", len(probs), env.NumActions())
	}
	var sum float64
	for _, x := range probs {
		sum += x
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("policy probs do not sum to 1: %v", probs)
	}

	a := NewA2C(env, 16, 0.02, 0.99, 1)
	if a.Lambda != 0.95 || a.ValueCoef != 0.5 || a.EntropyCoef != 0.01 || a.RolloutSteps != 256 {
		t.Fatalf("unexpected A2C defaults: %+v", a)
	}
	a2 := NewA2C(env, 16, 0.02, 0.99, 1, WithA2CLambda(3), WithA2CValueCoef(-1), WithA2CEntropyCoef(-1), WithA2CRolloutSteps(0))
	if a2.Lambda != 0.95 || a2.ValueCoef != 0.5 || a2.EntropyCoef != 0.01 || a2.RolloutSteps != 256 {
		t.Fatalf("invalid A2C options changed defaults: %+v", a2)
	}
}

// gradOf runs lossFn on a fresh tape and returns the flattened gradient of the
// scalar loss w.r.t. each parameter tensor.
func gradOf(t *testing.T, lossFn func(*backend.Context) (*tensor.Tensor, error), params []*tensor.Tensor) [][]float64 {
	t.Helper()
	tape := autograd.NewTape()
	loss, err := lossFn(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	out := make([][]float64, len(params))
	for pi, p := range params {
		g := tape.Grad(p)
		out[pi] = make([]float64, p.Numel())
		if g == nil {
			continue
		}
		for i := range out[pi] {
			out[pi][i] = g.AtF64(tensor.Unravel(i, p.Shape())...)
		}
	}
	return out
}

// ExamplePPO trains a PPO agent on the Chain MDP and confirms the learned policy
// prefers moving right (the optimal action) from the start state.
func ExamplePPO() {
	env := NewChain(5, 20)
	agent := NewPPO(env, 16, 0.02, 0.99, 7, WithPPORolloutSteps(200))
	for range 60 {
		if _, err := agent.Update(env); err != nil {
			panic(err)
		}
	}
	start := []float64{0, 0, 1, 0, 0} // middle state
	probs, err := agent.ActionProbs(start)
	if err != nil {
		panic(err)
	}
	if _, err := agent.Value(start); err != nil { // learned state value Vφ(start)
		panic(err)
	}
	fmt.Println("prefers right:", probs[1] > probs[0])
	// Output: prefers right: true
}

// ExampleA2C trains the A2C baseline on the Chain MDP and confirms it too learns to
// prefer the optimal right action from the start state.
func ExampleA2C() {
	env := NewChain(5, 20)
	agent := NewA2C(env, 16, 0.02, 0.99, 7, WithA2CRolloutSteps(200))
	for range 120 {
		if _, err := agent.Update(env); err != nil {
			panic(err)
		}
	}
	start := []float64{0, 0, 1, 0, 0}
	probs, err := agent.ActionProbs(start)
	if err != nil {
		panic(err)
	}
	if _, err := agent.Value(start); err != nil { // learned state value Vφ(start)
		panic(err)
	}
	fmt.Println("prefers right:", probs[1] > probs[0])
	// Output: prefers right: true
}
