package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §T440: γ=1 collapses CFG to the ordinary conditional distribution, so greedy CFGDecode must
// equal the model's own greedy Generate token for token (log-softmax preserves the argmax).
func TestCFGDecodeGammaOneIsPlainGenerate(t *testing.T) {
	model, prompt := loadGPTModel(t)
	neg := prompt[:2]
	maxNew := model.Config.Ctx - len(prompt)
	got, err := nlp.CFGDecode(model, prompt, neg, maxNew, 1.0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: CFG γ=1 %d != plain %d", i, got[i], want[i])
		}
	}
}

// §T440: γ=0 collapses CFG to the unconditional branch — greedy CFGDecode's generated tokens
// must equal greedy Generate seeded with the NEGATIVE prompt (both streams see the same
// appended tokens, so the recursion matches exactly).
func TestCFGDecodeGammaZeroFollowsNegativeBranch(t *testing.T) {
	model, prompt := loadGPTModel(t)
	neg := prompt[:2]
	maxNew := model.Config.Ctx - len(prompt)
	got, err := nlp.CFGDecode(model, prompt, neg, maxNew, 0.0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.Generate(neg, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	g, w := got[len(prompt):], want[len(neg):]
	if len(g) != len(w) {
		t.Fatalf("generated length %d vs %d", len(g), len(w))
	}
	for i := range w {
		if g[i] != w[i] {
			t.Fatalf("gen[%d]: CFG γ=0 %d != unconditional %d", i, g[i], w[i])
		}
	}
}

// §T440: empty prompts are rejected — the unconditional stream needs at least one token.
func TestCFGDecodeRejectsEmptyPrompts(t *testing.T) {
	model, prompt := loadGPTModel(t)
	if _, err := nlp.CFGDecode(model, prompt, nil, 2, 1.5, nlp.Greedy()); err == nil {
		t.Fatal("empty negPrompt accepted")
	}
	if _, err := nlp.CFGDecode(model, nil, prompt, 2, 1.5, nlp.Greedy()); err == nil {
		t.Fatal("empty prompt accepted")
	}
}
