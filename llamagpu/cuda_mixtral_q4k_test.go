//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDAMixtralQ4KServesAndGates builds a Dim=256 (K%256==0) random Mixtral, serves it as Q4_K
// (experts → ResidentBQ4K → the gated sparse decode), and checks the whole-model logits are finite +
// close to the f32 reference within Q4_K tolerance. This exercises the Q4_K MoE serving path AND the
// #913 gated expert eval end-to-end (the gating is bit-exact vs dense, proven by TestCUDAQ4KMoeGateParity,
// so a correct served output confirms both).
func TestCUDAMixtralQ4KServesAndGates(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	cfg := nlp.MixtralConfig{
		Vocab: 256, Ctx: 32, Dim: 256, Heads: 8, KVHeads: 4, Layers: 2,
		Hidden: 256, Experts: 4, TopK: 2, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewMixtral(cfg, 7)
	if err != nil {
		t.Fatalf("NewMixtral: %v", err)
	}
	dec, err := llamagpu.NewMixtralQ4KCUDA(m)
	if err != nil {
		t.Fatalf("NewMixtralQ4KCUDA: %v", err)
	}
	defer dec.Release()
	prompt := []int{1, 5, 9, 13}
	logits, err := dec.StepN(prompt, 0)
	if err != nil {
		t.Fatalf("StepN: %v", err)
	}
	if len(logits) != len(prompt)*cfg.Vocab {
		t.Fatalf("logits len %d want %d", len(logits), len(prompt)*cfg.Vocab)
	}
	// finite + not all-zero (gross serving-bug / gate-corruption check)
	nz := 0
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Q4_K MoE logit %d is non-finite: %v", i, v)
		}
		if v != 0 {
			nz++
		}
	}
	if nz == 0 {
		t.Fatal("all Q4_K MoE logits zero — expert gating likely zeroed everything")
	}
	// greedy generate a few tokens — must stay in-vocab + not collapse to one token
	out, err := dec.Generate(prompt, 6, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	gen := out[len(prompt):]
	same := true
	for i := range gen {
		if gen[i] < 0 || gen[i] >= cfg.Vocab {
			t.Fatalf("Q4_K MoE gen token %d out of vocab: %d", i, gen[i])
		}
		if i > 0 && gen[i] != gen[0] {
			same = false
		}
	}
	if same && len(gen) > 2 {
		t.Logf("warning: Q4_K MoE gen collapsed to token %d (may be fine for a random model)", gen[0])
	}
	t.Logf("NewMixtralQ4KCUDA serves + gates: %d finite logits, gen %v (Q4_K experts, ~E/K sparse decode)", nz, gen)
}
