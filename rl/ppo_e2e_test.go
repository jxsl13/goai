package rl

import (
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

// §T495: PPO closes the classic-RL loop — an actor-critic on the Chain environment built from
// the package's own GAE and nn.PPOClipLoss, the last unconnected pair (GAE existed as a pure
// function, PPOClip as a loss; neither had an agent driving them). Per iteration: collect
// episodes with the current policy, compute GAE advantages from the critic's values, then take
// clipped policy steps and value-regression steps. The late-phase average return must reach the
// near-optimal level the REINFORCE test requires, and the critic's value at the start state must
// approach the discounted optimal return.
func TestPPOLearnsChain(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a PPO agent; skipped in -short")
	}
	env := NewChain(5, 20)
	const (
		gamma  = 0.99
		lam    = 0.95
		iters  = 60
		epsPPO = 0.2
	)
	rng := rand.New(rand.NewPCG(42, 0x990))

	actor := policyNet(env.ObsDim(), 16, env.NumActions(), 7)
	critic := nn.NewSequential(
		nn.NewLinear(tensor.F64, env.ObsDim(), 16, 8), nn.ReLU(),
		nn.NewLinear(tensor.F64, 16, 1, 9),
	)
	aOpt := nn.NewAdamW(actor.Params(), 0.02, 0)
	cOpt := nn.NewAdamW(critic.Params(), 0.02, 0)

	obsT := func(states [][]float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{len(states), env.ObsDim()})
		for i, s := range states {
			for j, v := range s {
				x.SetF64(v, i, j)
			}
		}
		return x
	}
	values := func(states [][]float64) []float64 {
		v, err := critic.Forward(backend.NewContext(), obsT(states))
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float64, len(states))
		for i := range out {
			out[i] = v.AtF64(i, 0)
		}
		return out
	}

	var early, late float64
	for it := range iters {
		// 1. collect 8 episodes with the current policy.
		var states [][]float64
		var actions []int
		var logpOld, rewards []float64
		var dones []bool
		var iterReturn float64
		for range 8 {
			obs := env.Reset()
			for {
				logits, err := actor.Forward(backend.NewContext(), obsT([][]float64{obs}))
				if err != nil {
					t.Fatal(err)
				}
				// softmax sample + logp
				var mx float64 = math.Inf(-1)
				for a := range env.NumActions() {
					if x := logits.AtF64(0, a); x > mx {
						mx = x
					}
				}
				var sum float64
				probs := make([]float64, env.NumActions())
				for a := range env.NumActions() {
					probs[a] = math.Exp(logits.AtF64(0, a) - mx)
					sum += probs[a]
				}
				u := rng.Float64() * sum
				act := 0
				cum := 0.0
				for a, p := range probs {
					cum += p
					if u <= cum {
						act = a
						break
					}
				}
				states = append(states, obs)
				actions = append(actions, act)
				logpOld = append(logpOld, math.Log(probs[act]/sum))
				next, rew, done := env.Step(act)
				rewards = append(rewards, rew)
				dones = append(dones, done)
				iterReturn += rew
				obs = next
				if done {
					break
				}
			}
		}
		iterReturn /= 8
		if it < 10 {
			early += iterReturn / 10
		}
		if it >= iters-10 {
			late += iterReturn / 10
		}

		// 2. GAE advantages + returns from the critic's values.
		adv, rets := GAE(rewards, values(states), dones, 0, true, gamma, lam)
		// normalize advantages (standard PPO practice).
		var m, s float64
		for _, a := range adv {
			m += a
		}
		m /= float64(len(adv))
		for _, a := range adv {
			s += (a - m) * (a - m)
		}
		s = math.Sqrt(s/float64(len(adv))) + 1e-8
		advT := tensor.New(tensor.F64, tensor.Shape{len(adv)})
		for i, a := range adv {
			advT.SetF64((a-m)/s, i)
		}
		oldT := tensor.New(tensor.F64, tensor.Shape{len(logpOld)})
		for i, v := range logpOld {
			oldT.SetF64(v, i)
		}

		// 3. clipped policy updates (4 epochs over the batch).
		for range 4 {
			tape := autograd.NewTape()
			ctx := tape.Context()
			logits, err := actor.Forward(ctx, obsT(states))
			if err != nil {
				t.Fatal(err)
			}
			lp, err := nn.TokenLogProbs(ctx, logits, actions)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.PPOClipLoss(ctx, lp, oldT, advT, epsPPO)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := aOpt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
		// 4. critic regression toward the GAE returns.
		retT := tensor.New(tensor.F64, tensor.Shape{len(rets), 1})
		for i, r := range rets {
			retT.SetF64(r, i, 0)
		}
		for range 4 {
			tape := autograd.NewTape()
			ctx := tape.Context()
			v, err := critic.Forward(ctx, obsT(states))
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.MSE(ctx, v, retT)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := cOpt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Logf("PPO on Chain: early avg return %.3f → late %.3f", early, late)
	if late < 0.9 {
		t.Fatalf("PPO did not learn: late avg %.3f", late)
	}
	if late <= early {
		t.Fatalf("no improvement: %.3f → %.3f", early, late)
	}
	// critic sanity: value of the start state approaches the optimal discounted return.
	v0 := values([][]float64{env.Reset()})[0]
	optimal := math.Pow(gamma, 4) // 4 steps right to the terminal reward 1
	t.Logf("critic V(start) = %.3f (optimal ≈ %.3f)", v0, optimal)
	if math.Abs(v0-optimal) > 0.35 {
		t.Fatalf("critic value %.3f far from the optimal %.3f", v0, optimal)
	}
}
