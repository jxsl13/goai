package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §T436: SampleWithHistory wires ApplyPenalties into the Sampler — the CTRL repeat penalty
// steers GREEDY decoding away from a token already in the history (penalties act before the
// greedy/temperature split), and the caller's logits are never mutated.
func TestSamplerRepeatPenaltyGreedy(t *testing.T) {
	logits := []float64{2.0, 1.5, 0.1}
	if got := nlp.Greedy().SampleWithHistory(logits, nil); got != 0 {
		t.Fatalf("no history: want argmax 0, got %d", got)
	}
	s := nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithRepeatPenalty(2))
	if got := s.SampleWithHistory(logits, []int{0}); got != 1 {
		t.Fatalf("penalized 0 (2.0/2=1.0 < 1.5): want 1, got %d", got)
	}
	if logits[0] != 2.0 {
		t.Fatalf("SampleWithHistory mutated the caller's logits: %v", logits)
	}
}

// §T436: frequency penalty distinguishes counts where presence cannot.
func TestSamplerFrequencyVsPresence(t *testing.T) {
	logits := []float64{1.0, 0.85, 0.0}
	hist := []int{0, 0, 0, 1} // token0 3×, token1 1×

	// freq 0.1: token0 1.0−0.3=0.70 < token1 0.85−0.1=0.75 → count flips the argmax.
	s := nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithFrequencyPenalty(0.1))
	if got := s.SampleWithHistory(logits, hist); got != 1 {
		t.Fatalf("frequency: want 1, got %d", got)
	}
	// presence 0.2 hits both once: 0.8 vs 0.65 → argmax unchanged (count-blind).
	s = nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithPresencePenalty(0.2))
	if got := s.SampleWithHistory(logits, hist); got != 0 {
		t.Fatalf("presence: want 0, got %d", got)
	}
	// presence 1.2 pushes both seen tokens below the unseen token2.
	s = nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithPresencePenalty(1.2))
	if got := s.SampleWithHistory(logits, hist); got != 2 {
		t.Fatalf("presence 1.2: want 2, got %d", got)
	}
}

// §T436: WithPenaltyWindow — occurrences older than the last-N window are invisible.
func TestSamplerPenaltyWindow(t *testing.T) {
	logits := []float64{1.0, 0.9}
	s := nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithRepeatPenalty(3), nlp.WithPenaltyWindow(2))
	if got := s.SampleWithHistory(logits, []int{0, 1, 1}); got != 0 {
		t.Fatalf("token 0 outside the 2-token window must be unpenalized: want 0, got %d", got)
	}
	if got := s.SampleWithHistory(logits, []int{1, 0, 0}); got != 1 {
		t.Fatalf("token 0 inside the window, 1 outside it (1.0/3 < 0.9): want 1, got %d", got)
	}
}

// §T436 end-to-end: with an overwhelming presence penalty, greedy Generate can never emit a
// token that is already in the running sequence (prompt included) — every step must pick a fresh
// token. This proves the history actually flows through the Generate loop into the sampler.
func TestGenerateWithPenaltyNeverRepeats(t *testing.T) {
	model, prompt := loadGPTModel(t)
	maxNew := model.Config.Ctx - len(prompt)
	if v := model.Config.Vocab - len(prompt); v < maxNew {
		maxNew = v
	}
	if maxNew < 2 {
		t.Skip("test model too small")
	}
	s := nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithPresencePenalty(1e9))
	out, err := model.Generate(prompt, maxNew, s)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[int]bool{}
	for _, tok := range prompt {
		seen[tok] = true
	}
	for i, tok := range out[len(prompt):] {
		if seen[tok] {
			t.Fatalf("step %d repeated token %d despite an overwhelming presence penalty — history not applied", i, tok)
		}
		seen[tok] = true
	}
	t.Logf("%d generated tokens, all distinct from the full history", maxNew)
}

// §T436: with no penalties configured, SampleWithHistory is exactly Sample — same rng stream,
// so stochastic generation is unchanged for existing callers.
func TestSampleWithHistoryNoPenaltiesIdentical(t *testing.T) {
	logits := []float64{0.3, 0.2, 0.9, 0.1, 0.5}
	a := nlp.NewSampler(42, nlp.WithTemperature(0.8), nlp.WithTopK(3))
	b := nlp.NewSampler(42, nlp.WithTemperature(0.8), nlp.WithTopK(3))
	for i := range 50 {
		if x, y := a.Sample(logits), b.SampleWithHistory(logits, []int{0, 1, 2}); x != y {
			t.Fatalf("draw %d: Sample %d != SampleWithHistory %d", i, x, y)
		}
	}
}
