package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §T439: Watermark.Sampler plugs watermarking into the real Generate loop. With an
// overwhelming δ and a greedy inner sampler, EVERY generated token must be green w.r.t.
// its predecessor — checked directly against GreenMask, and confirmed by the detector.
func TestWatermarkSamplerThroughGenerate(t *testing.T) {
	model, prompt := loadGPTModel(t)
	prompt = prompt[:1] // leave room: the z-test needs enough scored tokens
	maxNew := model.Config.Ctx - len(prompt)
	w, err := nlp.NewWatermark(model.Config.Vocab, nlp.WithWatermarkDelta(1000))
	if err != nil {
		t.Fatal(err)
	}
	out, err := model.Generate(prompt, maxNew, w.Sampler(nlp.Greedy()))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("generated %d, want %d", len(out), len(prompt)+maxNew)
	}
	for i := len(prompt); i < len(out); i++ {
		if !w.GreenMask(out[i-1])[out[i]] {
			t.Fatalf("token %d (%d) is not green for predecessor %d — bias did not reach the loop", i, out[i], out[i-1])
		}
	}
	z, green, scored := w.Detect(out[len(prompt)-1:])
	if green != scored {
		t.Fatalf("detector: %d/%d green, want all", green, scored)
	}
	if !w.IsWatermarked(out[len(prompt)-1:]) {
		t.Fatalf("detector rejected a fully-green sequence (z=%.2f, %d scored)", z, scored)
	}
	t.Logf("watermarked generate: %d tokens, all green, z=%.2f", scored, z)
}

// §T439: a plain (unwatermarked) generation from the same model must NOT be detected —
// the detector's false-positive guard on real model output.
func TestWatermarkDetectorRejectsPlainGenerate(t *testing.T) {
	model, prompt := loadGPTModel(t)
	prompt = prompt[:1]
	maxNew := model.Config.Ctx - len(prompt)
	out, err := model.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	w, err := nlp.NewWatermark(model.Config.Vocab, nlp.WithWatermarkDelta(1000))
	if err != nil {
		t.Fatal(err)
	}
	if w.IsWatermarked(out[len(prompt)-1:]) {
		t.Fatal("plain greedy output flagged as watermarked")
	}
}

// §T439: RegexGuide.Sampler enforces the FSM through the real Generate loop: with the
// pattern (ab)+ over a vocab where token 0 = "a" and token 1 = "b", the generated ids
// must alternate 0,1,0,1,… no matter what the model prefers.
func TestGuidedSamplerThroughGenerate(t *testing.T) {
	model, prompt := loadGPTModel(t)
	vocab := make([]string, model.Config.Vocab)
	vocab[0], vocab[1] = "a", "b"
	for i := 2; i < len(vocab); i++ {
		vocab[i] = "z" // never matches the pattern
	}
	g, err := nlp.NewRegexGuide("(ab)+", vocab)
	if err != nil {
		t.Fatal(err)
	}
	maxNew := model.Config.Ctx - len(prompt)
	if maxNew%2 == 1 {
		maxNew-- // even length so the run ends in an accepting state
	}
	out, err := model.Generate(prompt, maxNew, g.Sampler(nlp.Greedy(), -1))
	if err != nil {
		t.Fatal(err)
	}
	gen := out[len(prompt):]
	if len(gen) != maxNew {
		t.Fatalf("generated %d, want %d", len(gen), maxNew)
	}
	for i, tok := range gen {
		if want := i % 2; tok != want {
			t.Fatalf("position %d: got token %d, want %d — the guide did not constrain the loop (gen=%v)", i, tok, want, gen)
		}
	}
	t.Logf("guided generate: %v (strictly (ab)+)", gen)
}
