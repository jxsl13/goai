//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDAJambaMatchesReference is the parity anchor for GoAI's FIRST HYBRID GPU decoder: an nlp.Jamba
// interleaves Mamba selective-scan mixers with NoPE grouped-query attention, and per layer runs a
// sparse-MoE or a dense SwiGLU FFN. The GPU decoder dispatches each layer to the right mixer (Mamba
// recurrence carrying conv/SSM state, or attention over a growing KV cache) then the right FFN, and
// must reproduce the reference Jamba.Forward logits at EVERY prompt position. Comparing every position
// (not just the last) proves the Mamba layers' state recurrence and the attention layers' KV cache stay
// in lockstep with the full-sequence reference across the whole run. Both Step and StepN are checked.
func TestCUDAJambaMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/jamba_hf.safetensors")
	if err != nil {
		t.Skipf("jamba testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.JambaFromHF(ts, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		t.Fatalf("JambaFromHF: %v", err)
	}

	dec, err := llamagpu.NewJambaCUDA(m)
	if err != nil {
		t.Fatalf("NewJambaCUDA: %v", err)
	}
	defer dec.Release()

	prompt := []int{3, 7, 1, 9, 4, 2, 8}
	refT, err := m.Forward(backend.NewContext().WithBackend(backend.Reference()), prompt)
	if err != nil {
		t.Fatalf("reference Forward: %v", err)
	}
	vocab := refT.Shape()[1]

	// Sequential decode: each Step carries the Mamba conv/SSM state + appends to the attention KV cache
	// (state reset at pos==0). Row i must match the reference logits after consuming token i.
	for pos, tok := range prompt {
		got, err := dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("cuda Step pos %d: %v", pos, err)
		}
		if len(got) != vocab {
			t.Fatalf("pos %d: got %d logits, want vocab %d", pos, len(got), vocab)
		}
		for j := range got {
			want := refT.AtF64(pos, j)
			if math.IsNaN(float64(got[j])) || math.Abs(float64(got[j])-want) > 2e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("jamba Step pos %d logit[%d]: cuda %v vs reference %v", pos, j, got[j], want)
			}
		}
	}

	// StepN prefill (loops Step for Jamba, since Mamba layers force sequential): every row must match.
	all, err := dec.StepN(prompt, 0)
	if err != nil {
		t.Fatalf("cuda StepN: %v", err)
	}
	if len(all) != len(prompt)*vocab {
		t.Fatalf("StepN returned %d logits, want %d", len(all), len(prompt)*vocab)
	}
	for i := range prompt {
		for j := 0; j < vocab; j++ {
			want := refT.AtF64(i, j)
			got := all[i*vocab+j]
			if math.IsNaN(float64(got)) || math.Abs(float64(got)-want) > 2e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("jamba StepN row %d logit[%d]: cuda %v vs reference %v", i, j, got, want)
			}
		}
	}
	t.Logf("llamagpu NewJambaCUDA matches reference Jamba.Forward at all %d positions (Step + StepN); first hybrid (Mamba+attention+MoE) GPU decoder", len(prompt))
}

// TestCUDAJambaGenerateReuseSafe checks the serving-safety property of the hybrid decoder: a second
// Generate on the SAME decoder must produce the same tokens as the first (and as a freshly-built
// decoder), i.e. Generate's StepN(_,0) prefill fully resets the carried state. This is stronger for
// Jamba than for a pure transformer: the Mamba layers hold conv/SSM state across Step, so if the pos==0
// reset missed them (or wrongly reset an attention layer), the reused run would drift from the fresh one.
// Greedy (temperature 0) makes the run deterministic, so equality is exact.
func TestCUDAJambaGenerateReuseSafe(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/jamba_hf.safetensors")
	if err != nil {
		t.Skipf("jamba testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.JambaFromHF(ts, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		t.Fatalf("JambaFromHF: %v", err)
	}
	prompt := []int{5, 2, 9, 1}
	const maxNew = 6

	dec, err := llamagpu.NewJambaCUDA(m)
	if err != nil {
		t.Fatalf("NewJambaCUDA: %v", err)
	}
	defer dec.Release()

	first, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate #1: %v", err)
	}
	reused, err := dec.Generate(prompt, maxNew, nlp.Greedy()) // same decoder: must reset conv/SSM + KV
	if err != nil {
		t.Fatalf("Generate #2 (reuse): %v", err)
	}
	// A fresh decoder is the ground truth for "no carried state".
	fresh, err := llamagpu.NewJambaCUDA(m)
	if err != nil {
		t.Fatalf("NewJambaCUDA (fresh): %v", err)
	}
	defer fresh.Release()
	freshOut, err := fresh.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate (fresh): %v", err)
	}

	if len(first) != len(prompt)+maxNew {
		t.Fatalf("Generate returned %d tokens, want %d", len(first), len(prompt)+maxNew)
	}
	for i := range first {
		if reused[i] != first[i] {
			t.Fatalf("reuse diverged at token %d: reused %d vs first %d — pos==0 did not fully reset hybrid state", i, reused[i], first[i])
		}
		if freshOut[i] != first[i] {
			t.Fatalf("fresh decoder disagrees at token %d: fresh %d vs first %d", i, freshOut[i], first[i])
		}
	}
	t.Logf("NewJambaCUDA Generate is reuse-safe: %v (identical across reuse + fresh decoder)", first)
}
