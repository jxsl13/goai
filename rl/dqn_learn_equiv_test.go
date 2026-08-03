package rl

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
)

// TestDQNLearnSingleForwardMatchesDouble pins that reading the target's non-taken entries from the
// TAPED forward gives the same values the separate untaped forward gave.
//
// The claim is exact: a recorder appends nodes as ops execute and does not change execution, and no
// optimizer step ran between the two forwards, so both saw identical weights over identical states.
// The test computes the target both ways from one agent's state and compares raw bits.
func TestDQNLearnSingleForwardMatchesDouble(t *testing.T) {
	const obs, hidden, actions, n = 12, 32, 2, 24
	env := NewChain(obs, 20)
	d := NewDQN(env, hidden, 1e-3, 0.99, 5)
	rng := rand.New(rand.NewPCG(3, 4))
	batch := make([]transition, n)
	states := make([][]float64, n)
	for i := range batch {
		s := make([]float64, obs)
		for j := range s {
			s[j] = rng.NormFloat64()
		}
		batch[i] = transition{s: s, s2: s, a: rng.IntN(actions), r: rng.NormFloat64()}
		states[i] = s
	}

	// The two forwards the old code performed: one untaped, one taped.
	untaped, err := forward(backend.NewContext(), d.Net, states)
	if err != nil {
		t.Fatal(err)
	}
	tape := autograd.NewTape()
	taped, err := forward(tape.Context(), d.Net, states)
	if err != nil {
		t.Fatal(err)
	}
	// The online and target nets must be DISTINGUISHABLE here, or the comparison below would hold
	// for the wrong reason. NewDQN seeds both identically, so the target is perturbed first.
	for _, p := range d.Target.Params() {
		st := p.Storage().F64() // rank-agnostic: params here are F64, and biases are rank 1
		st[0] += 0.5
	}
	tgt, err := forward(backend.NewContext(), d.Target, states)
	if err != nil {
		t.Fatal(err)
	}
	same := true
	for i := range n {
		for a := range actions {
			if untaped.AtF64(i, a) != tgt.AtF64(i, a) {
				same = false
			}
		}
	}
	if same {
		t.Fatal("the online and target nets produce identical values, so this fixture cannot tell " +
			"which net a target row came from")
	}

	for i := range n {
		for a := range actions {
			u, tp := untaped.AtF64(i, a), taped.AtF64(i, a)
			if math.Float64bits(u) != math.Float64bits(tp) {
				t.Fatalf("[%d,%d]: untaped %v, taped %v — the tape changed the value, so the "+
					"second forward was not redundant", i, a, u, tp)
			}
		}
	}
}
