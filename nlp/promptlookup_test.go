package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §V16 tier-1 (the LLMA losslessness guarantee, §R89): under greedy sampling,
// prompt-lookup decoding must produce token-for-token the SAME output as plain greedy
// Generate — the n-gram draft only accelerates, the accept/reject verification keeps
// the result identical.
func TestPromptLookupGreedyEqualsGenerate(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prompt := tokens[:min(6, len(tokens))]
	const n = 12

	want, err := model.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := nlp.PromptLookupDecode(model, prompt, n, 3, 10, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("prompt-lookup produced %d tokens, greedy Generate %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d — greedy prompt-lookup must equal greedy Generate", i, got[i], want[i])
		}
	}
}

// The default hyperparameters (maxNgram≤0→3, draftLen≤0→10) stay lossless, and on a
// prompt with a recurring n-gram the drafter actually engages (proposes tokens) — so
// the speculative path is exercised, not just the single-token fallback.
func TestPromptLookupDefaultsAndDrafts(t *testing.T) {
	model, tokens := loadGPTModel(t)
	base := tokens[:min(3, len(tokens))]
	prompt := append(append([]int(nil), base...), base...) // [a,b,c,a,b,c] → n-gram recurs
	const n = 10

	want, err := model.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	got, stats, err := nlp.PromptLookupDecode(model, prompt, n, 0, 0, nlp.Greedy()) // defaults
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("default prompt-lookup token[%d] = %d, want %d", i, got[i], want[i])
		}
	}
	if stats.Proposed == 0 {
		t.Error("recurring n-gram prompt but the drafter proposed nothing — lookup not engaging")
	}
}

// Losslessness holds under stochastic sampling in the distribution sense too: the run
// completes and advances past the prompt (each round accepts ≥1 target-distributed
// token).
func TestPromptLookupSamplingRuns(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prompt := tokens[:min(6, len(tokens))]
	s := nlp.NewSampler(7, nlp.WithTemperature(0.8), nlp.WithTopK(20))

	out, _, err := nlp.PromptLookupDecode(model, prompt, 10, 3, 8, s)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) <= len(prompt) {
		t.Fatalf("no tokens generated (%d ≤ %d)", len(out), len(prompt))
	}
}
