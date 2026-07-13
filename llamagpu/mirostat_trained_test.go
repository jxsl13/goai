//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T473: Mirostat's defining claim — it CONTROLS the generated text's surprise (per-token
// cross-entropy) toward a target τ — measured on a REAL trained model for the first time.
// The measurement exposed the honest shape of that claim: Mirostat is a ONE-SIDED control. It
// truncates high-surprise tokens, so it can pull realized surprise DOWN to a τ below the model's
// natural entropy — but it cannot push surprise ABOVE the model's own distribution (with τ higher
// than the entropy, μ grows, truncation vanishes, and realized surprise saturates at plain
// full-distribution sampling; raising surprise needs temperature, which reshapes the
// distribution). Asserted accordingly: τ below entropy is tracked; τ above entropy saturates at
// the untruncated baseline; a hot fixed temperature exceeds both.
func TestMirostatControlsSurpriseOnTrainedModel(t *testing.T) {
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
	cfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}
	m := trainableGPT(t, cfg, 1)
	tf, tl := trainCharGPT(t, m, be, toks, 240, 128, 3e-3)
	t.Logf("model: CE %.3f → %.3f", tf, tl)

	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	prompt := toks[:16]
	const maxNew = 200

	// generate with sampler s, then measure the REALIZED surprise −log₂ p(token) of every
	// generated token under the model itself (via the decoder's own step logits).
	meanSurprise := func(s nlp.TokenSampler) float64 {
		out, err := dec.Generate(prompt, maxNew, s)
		if err != nil {
			t.Fatal(err)
		}
		var bits float64
		pos := 0
		var logits []float32
		for _, tok := range out[:len(out)-1] {
			l, err := dec.Step(tok, pos)
			if err != nil {
				t.Fatal(err)
			}
			logits = l
			pos++
			if pos >= len(prompt) { // score the token GENERATED at this position
				next := out[pos]
				var mx float64 = math.Inf(-1)
				for _, v := range logits {
					if float64(v) > mx {
						mx = float64(v)
					}
				}
				var sum float64
				for _, v := range logits {
					sum += math.Exp(float64(v) - mx)
				}
				p := math.Exp(float64(logits[next])-mx) / sum
				bits += -math.Log2(p)
			}
		}
		return bits / float64(maxNew)
	}

	sLow := meanSurprise(nlp.NewMirostat(7, nlp.WithMirostatTau(0.5)))
	sHigh := meanSurprise(nlp.NewMirostat(7, nlp.WithMirostatTau(4.0)))
	base := meanSurprise(nlp.NewSampler(7, nlp.WithTemperature(1))) // untruncated model entropy
	hot := meanSurprise(nlp.NewSampler(7, nlp.WithTemperature(1.8)))

	t.Logf("realized mean surprise: mirostat τ=0.5 → %.2f | τ=4 → %.2f | plain temp1 → %.2f | temp1.8 → %.2f bits",
		sLow, sHigh, base, hot)
	// (1) a target BELOW the model's entropy is tracked from above.
	if sLow > 1.0 || sLow >= base-0.3 {
		t.Fatalf("mirostat τ=0.5: realized %.2f — should be pulled well below the untruncated %.2f toward the target", sLow, base)
	}
	// (2) a target ABOVE the model's entropy saturates at untruncated sampling — NOT at τ.
	if math.Abs(sHigh-base) > 0.5 {
		t.Fatalf("mirostat τ=4: realized %.2f should saturate near the untruncated %.2f (one-sided control)", sHigh, base)
	}
	if sHigh > 3.0 {
		t.Fatalf("mirostat τ=4 must not reach the target above the model's entropy (got %.2f)", sHigh)
	}
	// (3) raising surprise needs temperature, which Mirostat does not do.
	if hot <= base+0.5 {
		t.Fatalf("hot sampling (%.2f) should clearly exceed the model's natural %.2f — no contrast", hot, base)
	}
}
