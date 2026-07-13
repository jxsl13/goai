//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
)

// §T481: beam search's core promise measured on a REAL trained model — a width-4 beam must find a
// continuation whose total sequence log-probability is at least the greedy path's (greedy is the
// width-1 special case and is famously not the sequence argmax), and Diverse Beam Search must buy
// its diversity (distinct group outputs) at a bounded likelihood cost.
func TestBeamSearchOnTrainedModel(t *testing.T) {
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

	ctx := backend.NewContext().WithBackend(be)
	next := func(prefix []int) []float64 {
		logits, err := m.Forward(ctx, prefix)
		if err != nil {
			t.Fatal(err)
		}
		last := logits.Shape()[0] - 1
		out := make([]float64, vocab)
		for v := range vocab {
			out[v] = logits.AtF64(last, v)
		}
		return out
	}
	// sequence log-prob of a continuation under the model (log-softmax per step).
	seqLogp := func(seq []int, promptLen int) float64 {
		var lp float64
		for i := promptLen; i < len(seq); i++ {
			logits := next(seq[:i])
			var mx float64 = math.Inf(-1)
			for _, v := range logits {
				if v > mx {
					mx = v
				}
			}
			var sum float64
			for _, v := range logits {
				sum += math.Exp(v - mx)
			}
			lp += logits[seq[i]] - mx - math.Log(sum)
		}
		return lp
	}

	prompt := toks[:16]
	const maxNew = 24

	greedy, err := m.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	gLP := seqLogp(greedy, len(prompt))

	beams := nlp.BeamSearch(next, prompt, 4, maxNew, -1, 0)
	if len(beams) == 0 {
		t.Fatal("no beams returned")
	}
	bLP := seqLogp(beams[0].Tokens, len(prompt))
	t.Logf("sequence log-prob of %d tokens: greedy %.3f | beam-4 %.3f (Δ %.3f nats)", maxNew, gLP, bLP, bLP-gLP)
	if bLP < gLP-1e-9 {
		t.Fatalf("beam-4 found a WORSE sequence than greedy (%.4f < %.4f)", bLP, gLP)
	}
	// the reported score must agree with independent scoring (α=0: score == raw log-prob).
	if math.Abs(beams[0].Score-bLP) > 1e-6 {
		t.Fatalf("beam score %.6f disagrees with independent log-prob %.6f", beams[0].Score, bLP)
	}

	// Diverse Beam: groups must produce distinct sequences at bounded likelihood cost.
	div, err := nlp.DiverseBeamSearch(next, prompt, 4, 4, maxNew, -1, 0, 2.0)
	if err != nil {
		t.Fatal(err)
	}
	distinct := map[string]bool{}
	worst := math.Inf(1)
	for _, b := range div {
		key := ""
		for _, tok := range b.Tokens[len(prompt):] {
			key += string(rune('A' + tok))
		}
		distinct[key] = true
		if lp := seqLogp(b.Tokens, len(prompt)); lp < worst {
			worst = lp
		}
	}
	t.Logf("diverse beam (4 groups): %d distinct sequences, worst log-prob %.3f (best beam %.3f)", len(distinct), worst, bLP)
	if len(distinct) < 2 {
		t.Fatalf("diverse beam produced no diversity (%d distinct)", len(distinct))
	}
	if worst < bLP-float64(maxNew) { // ≥ e^-1 per token of the best — diversity, not garbage
		t.Fatalf("diversity cost unbounded: worst %.3f vs best %.3f over %d tokens", worst, bLP, maxNew)
	}
}
