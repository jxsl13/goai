package rl

import (
	"math/rand/v2"
	"testing"
)

// benchDQNLearn times one TD(0) step on a pre-filled replay buffer. `learn` runs on EVERY
// environment step inside Episode, so its cost is paid per step rather than per episode — which is
// why it deserves its own cell rather than being measured through Episode, where the environment
// and the epsilon-greedy action selection would dilute it.
//
// The buffer is filled directly rather than by running an environment, so the benchmark measures
// the learning step and nothing else, and the batch it samples is the same size a real run uses.
func benchDQNLearn(b *testing.B, obs, hidden, actions, bufN int) {
	// Chain sizes the net: ObsDim is its state count and NumActions is 2, so the caller must pass
	// the matching action count — a mismatch indexes past the net's output row.
	env := NewChain(obs, 20)
	if env.NumActions() != actions {
		b.Fatalf("fixture mismatch: env has %d actions, benchmark uses %d", env.NumActions(), actions)
	}
	d := NewDQN(env, hidden, 1e-3, 0.99, 7)
	rng := rand.New(rand.NewPCG(11, 22))
	d.buf = make([]transition, 0, bufN)
	for range bufN {
		s := make([]float64, obs)
		s2 := make([]float64, obs)
		for j := range s {
			s[j] = rng.NormFloat64()
			s2[j] = rng.NormFloat64()
		}
		d.buf = append(d.buf, transition{
			s: s, s2: s2, a: rng.IntN(actions), r: rng.NormFloat64(), done: rng.IntN(8) == 0,
		})
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := d.learn(actions); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDQNLearn(b *testing.B) { benchDQNLearn(b, 16, 64, 2, 512) }
