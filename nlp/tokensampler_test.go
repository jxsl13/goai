package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §T437: Mirostat satisfies TokenSampler and runs through the real Generate loop — before this,
// Mirostat.Sample existed but no generation path could accept it (the same orphan class as
// ApplyPenalties, §T436). Deterministic for a fixed seed, valid ids, correct length.
func TestGenerateWithMirostat(t *testing.T) {
	model, prompt := loadGPTModel(t)
	maxNew := model.Config.Ctx - len(prompt)
	if maxNew > 6 {
		maxNew = 6
	}
	if maxNew < 1 {
		t.Skip("test model context too small")
	}
	out1, err := model.Generate(prompt, maxNew, nlp.NewMirostat(7))
	if err != nil {
		t.Fatal(err)
	}
	if len(out1) != len(prompt)+maxNew {
		t.Fatalf("generated %d tokens, want %d", len(out1), len(prompt)+maxNew)
	}
	for _, tok := range out1 {
		if tok < 0 || tok >= model.Config.Vocab {
			t.Fatalf("invalid token %d", tok)
		}
	}
	out2, err := model.Generate(prompt, maxNew, nlp.NewMirostat(7))
	if err != nil {
		t.Fatal(err)
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatal("Mirostat generation with the same seed must be deterministic")
		}
	}
}
