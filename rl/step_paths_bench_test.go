package rl

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The per-environment-step paths, which neither BenchmarkRLForward (one batched forward)
// nor BenchmarkSoftUpdate covered. Both are dispatch-bound rather than FLOP-bound at these
// widths, so the metric that moves is dispatch count and allocations, not arithmetic.

func benchRolloutNets() (Env, *nn.Sequential, *nn.Sequential) {
	env := NewChain(9, 50)
	actor := policyNet(env.ObsDim(), 64, env.NumActions(), 1)
	critic := nn.NewSequential(
		nn.NewLinear(tensor.F64, env.ObsDim(), 64, 2),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, 64, 1, 3),
	)
	return env, actor, critic
}

// BenchmarkPPORollout times collection alone — rlRollout is called directly so the update
// phase (which is genuinely batched) cannot mask the per-step cost.
func BenchmarkPPORollout(b *testing.B) {
	env, actor, critic := benchRolloutNets()
	b.ReportAllocs()
	for b.Loop() {
		rng := rand.New(rand.NewPCG(4, 5))
		if _, err := rlRollout(env, actor, critic, rng, 256); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPPOUpdate is the end-to-end share, so the rollout improvement can be reported
// as a fraction of what a caller actually runs.
func BenchmarkPPOUpdate(b *testing.B) {
	env := NewChain(9, 50)
	b.ReportAllocs()
	for b.Loop() {
		p := NewPPO(env, 64, 3e-4, 0.99, 7)
		if _, err := p.Update(env); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDQNLearn isolates learn from env stepping and epsilon-greedy acting: the replay
// buffer is prefilled outside the timer, so what is timed is exactly one TD(0) step.
func BenchmarkDQNLearn(b *testing.B) {
	for _, bs := range []int{32, 128} {
		b.Run(rlBenchName(bs), func(b *testing.B) {
			env := NewChain(9, 50)
			d := NewDQN(env, 64, 1e-3, 0.99, 11)
			d.BatchSize = bs
			// Prefill with real transitions from the env so shapes and value ranges match
			// production; done outside the timer.
			obs := env.Reset()
			for range 4096 {
				a := d.rng.IntN(env.NumActions())
				next, rew, done := env.Step(a)
				d.buf = append(d.buf, transition{s: obs, s2: next, a: a, r: rew, done: done})
				obs = next
				if done {
					obs = env.Reset()
				}
			}
			k := env.NumActions()
			b.ReportAllocs()
			for b.Loop() {
				if err := d.learn(k); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func rlBenchName(bs int) string {
	switch bs {
	case 32:
		return "batch32"
	case 128:
		return "batch128"
	}
	return "batch"
}
