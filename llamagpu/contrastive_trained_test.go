//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T480: contrastive decoding (expert−amateur) measured on REAL trained models — the natural
// setup the method was designed for: expert = fully trained model, amateur = the same
// architecture stopped early. Assertions follow the MECHANISM, not vibes: (1) β=0 collapses CD
// to exactly the expert's own greedy decoding (sharp equivalence, now on trained models);
// (2) with β>0, CD's output must be MORE surprising to the amateur than the expert's greedy
// output is (it maximizes expert−amateur log-probability), while the α-plausibility constraint
// keeps its expert-surprise bounded near greedy.
func TestContrastiveDecodingOnTrainedModels(t *testing.T) {
	if testing.Short() {
		t.Skip("trains two models; skipped in -short")
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

	expert := trainableGPT(t, cfg, 1)
	ef, el := trainCharGPT(t, expert, be, toks, 240, 128, 3e-3)
	amateur := trainableGPT(t, cfg, 1) // same init, stopped early
	af, al := trainCharGPT(t, amateur, be, toks, 30, 128, 3e-3)
	t.Logf("expert CE %.3f → %.3f | amateur CE %.3f → %.3f", ef, el, af, al)

	prompt := toks[:16]
	const maxNew = 96
	ctx := backend.NewContext().WithBackend(be)

	surpriseUnder := func(m *nlp.GPT, seq []int) float64 {
		logits, err := m.Forward(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		sub, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{logits},
			backend.SliceAttrs{Axis: 0, Start: len(prompt) - 1, End: len(seq) - 1})
		if err != nil {
			t.Fatal(err)
		}
		lp, err := nn.TokenLogProbs(ctx, sub[0], seq[len(prompt):])
		if err != nil {
			t.Fatal(err)
		}
		var bits float64
		for i := range lp.Shape()[0] {
			bits += -lp.AtF64(i) / math.Ln2
		}
		return bits / float64(lp.Shape()[0])
	}

	greedy, err := expert.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}

	// (1) β=0: CD ≡ the expert's own decoding, token for token, on trained models.
	cd0, err := nlp.ContrastiveDecode(expert, amateur, prompt, maxNew, 0.1, 0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range greedy {
		if cd0[i] != greedy[i] {
			t.Fatalf("β=0 CD diverged from plain expert greedy at token %d", i)
		}
	}

	// (2) β>0: the contrast objective must show in the amateur's surprise.
	cd, err := nlp.ContrastiveDecode(expert, amateur, prompt, maxNew, 0.1, 0.5, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	expGreedy := surpriseUnder(expert, greedy)
	expCD := surpriseUnder(expert, cd)
	amaGreedy := surpriseUnder(amateur, greedy)
	amaCD := surpriseUnder(amateur, cd)
	t.Logf("surprise under expert: greedy %.3f | CD %.3f — under amateur: greedy %.3f | CD %.3f bits",
		expGreedy, expCD, amaGreedy, amaCD)
	if amaCD <= amaGreedy {
		t.Fatalf("CD output should be MORE surprising to the amateur (%.3f vs %.3f) — the contrast objective did not act", amaCD, amaGreedy)
	}
	if expCD > expGreedy+1.0 {
		t.Fatalf("α-plausibility failed to keep CD near the expert (%.3f vs greedy %.3f)", expCD, expGreedy)
	}
}
