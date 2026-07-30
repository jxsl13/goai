package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// Two edge cases of RegexGuide.MaskLogits that nothing covered.
//
// They used to fall out of Advance's own bounds check, which MaskLogits reached once per token. That
// call is now hoisted and the memo read directly, so both cases are EXPLICIT branches — and a branch
// no test reaches is one a later edit can silently invert. Verified by mutation: allowing
// out-of-range tokens through, and dropping the invalid-state arm, each left the whole package green
// before these tests existed.
func maskGuide(t *testing.T) (*nlp.RegexGuide, int) {
	t.Helper()
	vocab := []string{"a", "b", "c", "d"}
	g, err := nlp.NewRegexGuide(`a+`, vocab)
	if err != nil {
		t.Fatal(err)
	}
	return g, len(vocab)
}

// A caller may hand MaskLogits a logits slice longer than the vocabulary — a model whose head is
// padded past the tokenizer's real vocab. Those ids have no token, so they must be masked, never
// left samplable.
func TestMaskLogitsMasksBeyondVocab(t *testing.T) {
	g, v := maskGuide(t)
	logits := make([]float64, v+3)
	g.MaskLogits(g.Start(), logits, -1)
	for i := v; i < len(logits); i++ {
		if !math.IsInf(logits[i], -1) {
			t.Fatalf("logit %d is past the vocabulary (%d tokens) and must be masked, got %v", i, v, logits[i])
		}
	}
	// And the fixture must not be vacuous: something inside the vocabulary stays allowed.
	if math.IsInf(logits[0], -1) {
		t.Fatal(`"a" is the only legal first token of a+ and must remain allowed`)
	}
}

// An out-of-range state id is a dead state: nothing may be sampled from it.
func TestMaskLogitsInvalidStateMasksEverything(t *testing.T) {
	g, v := maskGuide(t)
	for _, state := range []int{-1, 1 << 20} {
		logits := make([]float64, v)
		g.MaskLogits(state, logits, -1)
		for i, x := range logits {
			if !math.IsInf(x, -1) {
				t.Fatalf("state %d is invalid so every token must be masked; logit %d is %v", state, i, x)
			}
		}
	}
}
