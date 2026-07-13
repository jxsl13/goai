package nlp_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §V16 tier-1: with draft==target and greedy sampling, speculative decoding must
// produce token-for-token the SAME output as plain greedy Generate — the lossless
// guarantee at its strongest (deterministic). It also must accept every drafted
// token (p==q ⇒ accept prob 1).
func TestSpeculativeDecodeGreedyEqualsGenerate(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prompt := tokens[:min(4, len(tokens))]
	const n = 8

	want, err := model.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	got, stats, err := nlp.SpeculativeDecode(model, model, prompt, n, 4, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("speculative produced %d tokens, greedy Generate %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d — greedy speculative must equal greedy Generate", i, got[i], want[i])
		}
	}
	if r := stats.AcceptanceRate(); r != 1 {
		t.Errorf("draft==target must accept all drafts, got rate %v (%d/%d)", r, stats.Accepted, stats.Proposed)
	}
}

// The losslessness holds under stochastic sampling too: with draft==target every
// proposed token is accepted (accept prob min(1, p/q)=1), so the run advances by a
// full lookahead+1 each round.
func TestSpeculativeDecodeAllAcceptedSampling(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prompt := tokens[:min(4, len(tokens))]
	s := nlp.NewSampler(42, nlp.WithTemperature(0.8), nlp.WithTopK(20))

	out, stats, err := nlp.SpeculativeDecode(model, model, prompt, 12, 4, s)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) <= len(prompt) {
		t.Fatalf("no tokens generated (%d)", len(out))
	}
	if stats.Proposed == 0 || stats.Accepted != stats.Proposed {
		t.Errorf("draft==target must accept all proposed tokens, got %d/%d", stats.Accepted, stats.Proposed)
	}
}

// SpecStats reports how many of the draft model's proposed tokens the target
// accepted; the acceptance rate is what turns one target forward per round into an
// end-to-end speedup.
func ExampleSpecStats() {
	s := nlp.SpecStats{Proposed: 10, Accepted: 8}
	fmt.Printf("accepted %d/%d = %.0f%%\n", s.Accepted, s.Proposed, s.AcceptanceRate()*100)
	// Output: accepted 8/10 = 80%
}
