//go:build darwin && cgo

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T474: watermark detection POWER at realistic parameters, on a real trained model. §T439
// proved the mechanics with an overwhelming δ=1000; here the paper-default soft watermark
// (δ=2, γ=0.25) runs on the trained char-GPT — deliberately a HARD case, since the grammar
// corpus is low-entropy (~1.8 bits/token) and the paper itself warns that low-entropy text
// carries watermarks poorly (near-deterministic tokens ignore the green-list bias). Measured:
// the detection z-score must GROW with generated length, plain (unwatermarked) sampling must
// never be flagged, and whether δ=2 crosses the z=4 threshold at 300 tokens is REPORTED as the
// finding (either outcome is informative on low-entropy text).
func TestWatermarkPowerOnTrainedModel(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal not registered")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.GPTConfig{Vocab: vocab, Ctx: 512, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}
	m := trainableGPT(t, cfg, 1)
	tf, tl := trainCharGPT(t, m, be, toks, 240, 128, 3e-3)
	t.Logf("model: CE %.3f → %.3f", tf, tl)

	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	prompt := toks[:16]

	w, err := nlp.NewWatermark(cfg.Vocab) // paper defaults: γ=0.25, δ=2
	if err != nil {
		t.Fatal(err)
	}
	gen := func(sampler nlp.TokenSampler, n int) []int {
		out, err := dec.Generate(prompt, n, sampler)
		if err != nil {
			t.Fatal(err)
		}
		return out[len(prompt)-1:] // predecessor-led slice for Detect
	}

	zAt := func(n int) float64 {
		seq := gen(w.Sampler(nlp.NewSampler(7, nlp.WithTemperature(1))), n)
		z, green, scored := w.Detect(seq)
		t.Logf("δ=2, %3d tokens: %d/%d green, z=%.2f", n, green, scored, z)
		return z
	}
	z50 := zAt(50)
	z150 := zAt(150)
	z300 := zAt(300)

	// (1) power grows with length.
	if !(z300 > z50) {
		t.Fatalf("detection power must grow with length: z(50)=%.2f, z(300)=%.2f", z50, z300)
	}
	// (2) the watermark leaves a real statistical trace even at δ=2 on low-entropy text.
	if z300 < 1.5 {
		t.Fatalf("no meaningful watermark trace at 300 tokens (z=%.2f)", z300)
	}
	// (3) false-positive guard: plain sampled text is never flagged.
	for _, n := range []int{50, 150, 300} {
		plain := gen(nlp.NewSampler(7, nlp.WithTemperature(1)), n)
		if w.IsWatermarked(plain) {
			t.Fatalf("plain %d-token generation flagged as watermarked", n)
		}
	}
	// (4) the finding: does the paper-default δ cross the decision threshold here?
	if z300 >= nlp.WatermarkZThreshold {
		t.Logf("FINDING: δ=2 IS detectable (z=%.2f ≥ %.0f) at 300 tokens despite the low-entropy corpus", z300, nlp.WatermarkZThreshold)
	} else {
		t.Logf("FINDING: δ=2 does NOT reach z=%.0f at 300 tokens (z=%.2f, z150=%.2f) — the paper's low-entropy caveat reproduced; higher δ or more text needed", nlp.WatermarkZThreshold, z300, z150)
	}
}
