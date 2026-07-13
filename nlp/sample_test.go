package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// Greedy picks the argmax deterministically, ignoring temperature/RNG.
func TestGreedySampler(t *testing.T) {
	s := nlp.Greedy()
	for range 5 {
		if got := s.Sample([]float64{0.1, 2.5, -1, 0.3}); got != 1 {
			t.Fatalf("greedy = %d, want 1", got)
		}
	}
}

// top-k restricts the support to the k highest-logit tokens.
func TestTopKSupport(t *testing.T) {
	// logits favor tokens 3 and 1; k=2 → only {1,3} may be sampled
	logits := []float64{0.0, 3.0, 0.5, 4.0, 1.0}
	s := nlp.NewSampler(7, nlp.WithTopK(2))
	seen := map[int]int{}
	for range 2000 {
		seen[s.Sample(logits)]++
	}
	for tok := range seen {
		if tok != 1 && tok != 3 {
			t.Errorf("top-k=2 sampled disallowed token %d", tok)
		}
	}
	if seen[3] == 0 || seen[1] == 0 {
		t.Error("both top-2 tokens should appear")
	}
}

// top-p nucleus keeps the smallest desc-prob prefix reaching p; the crossing
// token is included and at least the top token always survives.
func TestTopPNucleus(t *testing.T) {
	// probs ∝ softmax; token 0 dominates. With p=0.5 and a dominant token, only
	// the top token(s) needed to reach 0.5 are kept.
	logits := []float64{5.0, 3.0, 1.0, 0.0} // softmax ≈ [0.84,0.11,0.015,0.006]
	s := nlp.NewSampler(3, nlp.WithTopP(0.5))
	seen := map[int]bool{}
	for range 3000 {
		seen[s.Sample(logits)] = true
	}
	// top token prob ≈0.84 ≥0.5 already → nucleus = {0} only
	if len(seen) != 1 || !seen[0] {
		t.Errorf("top-p=0.5 nucleus = %v, want {0}", seen)
	}

	// p=0.95 → need tokens 0 and 1 (0.84+0.11=0.95)
	s2 := nlp.NewSampler(3, nlp.WithTopP(0.95))
	seen2 := map[int]bool{}
	for range 5000 {
		seen2[s2.Sample(logits)] = true
	}
	if seen2[2] || seen2[3] {
		t.Errorf("top-p=0.95 leaked low-prob tokens: %v", seen2)
	}
	if !seen2[0] || !seen2[1] {
		t.Errorf("top-p=0.95 should keep tokens 0 and 1: %v", seen2)
	}
}

// Sampling is deterministic under a fixed seed.
func TestSamplerDeterministic(t *testing.T) {
	logits := []float64{1, 2, 3, 0.5, -1}
	a := nlp.NewSampler(42, nlp.WithTemperature(0.8))
	b := nlp.NewSampler(42, nlp.WithTemperature(0.8))
	for range 100 {
		if a.Sample(logits) != b.Sample(logits) {
			t.Fatal("same seed must produce same samples")
		}
	}
}

// End-to-end greedy generation using the KV-cache: deterministic and the right
// length.
func TestGenerateGreedy(t *testing.T) {
	model, prompt := loadGPTModel(t)
	out1, err := model.Generate(prompt, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out1) != len(prompt)+3 {
		t.Fatalf("generated %d tokens, want %d", len(out1), len(prompt)+3)
	}
	// greedy is deterministic
	out2, _ := model.Generate(prompt, 3, nlp.Greedy())
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatal("greedy generation must be deterministic")
		}
	}
	// generated tokens must be valid ids
	for _, tok := range out1 {
		if tok < 0 || tok >= model.Config.Vocab {
			t.Fatalf("invalid token %d", tok)
		}
	}
}

// Temperature → 0 approaches greedy; large temperature broadens the support.
func TestTemperatureEffect(t *testing.T) {
	logits := []float64{0.2, 2.0, 0.1, 1.0, 0.3}
	cold := nlp.NewSampler(1, nlp.WithTemperature(0.01))
	seen := map[int]int{}
	for range 500 {
		seen[cold.Sample(logits)]++
	}
	if seen[1] < 480 { // near-greedy → almost always the argmax (token 1)
		t.Errorf("cold temperature not near-greedy: %v", seen)
	}
	hot := nlp.NewSampler(1, nlp.WithTemperature(5.0))
	seenHot := map[int]bool{}
	for range 2000 {
		seenHot[hot.Sample(logits)] = true
	}
	if len(seenHot) < 4 { // hot → most tokens appear
		t.Errorf("hot temperature too narrow: %v", seenHot)
	}
}
