//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
)

// §T482: DoLa layer-contrast decoding on the REAL trained model (§T480's discipline applied to
// the layer dimension): DoLa maximizes mature−premature log-probability within the mature
// plausibility set, so on a trained model its output must be MORE surprising to the early-exit
// (premature-layer) distribution than the plain greedy output is, while the α-plausibility
// constraint keeps mature-surprise bounded. α=1 must still collapse to exact greedy.
func TestDoLaOnTrainedModel(t *testing.T) {
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

	prompt := toks[:16]
	const maxNew = 48
	ctx := backend.NewContext().WithBackend(be)
	layers := []int{0} // premature candidate: after block 0

	// surprise (bits/token) of seq's generated tail under the MATURE head or the layer-0
	// EARLY-EXIT head.
	surprise := func(seq []int, premature bool) float64 {
		mature, prem, err := m.ForwardEarlyExit(ctx, seq, layers)
		if err != nil {
			t.Fatal(err)
		}
		logits := mature
		if premature {
			logits = prem[0]
		}
		var bits float64
		n := 0
		for i := len(prompt); i < len(seq); i++ {
			row := i - 1
			v := logits.Shape()[1]
			var mx float64 = math.Inf(-1)
			for j := range v {
				if x := logits.AtF64(row, j); x > mx {
					mx = x
				}
			}
			var sum float64
			for j := range v {
				sum += math.Exp(logits.AtF64(row, j) - mx)
			}
			p := math.Exp(logits.AtF64(row, seq[i])-mx) / sum
			bits += -math.Log2(p)
			n++
		}
		return bits / float64(n)
	}

	greedy, err := m.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}

	// α=1 collapse on the trained model.
	dola1, err := nlp.DoLaDecode(m, prompt, maxNew, layers, 1.0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range greedy {
		if dola1[i] != greedy[i] {
			t.Fatalf("α=1 DoLa diverged from greedy at token %d", i)
		}
	}

	// α=0.1: the layer-contrast objective must show, and the plausibility constraint's ACTUAL
	// per-token guarantee must hold: chosen prob ≥ α·max-prob ⇒ mean surprise ≤ greedy + log₂(1/α).
	dola, err := nlp.DoLaDecode(m, prompt, maxNew, layers, 0.1, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	matGreedy := surprise(greedy, false)
	matDoLa := surprise(dola, false)
	preGreedy := surprise(greedy, true)
	preDoLa := surprise(dola, true)
	t.Logf("surprise under mature: greedy %.3f | DoLa(α=0.1) %.3f — under layer-0 early exit: greedy %.3f | DoLa %.3f bits",
		matGreedy, matDoLa, preGreedy, preDoLa)
	if preDoLa <= preGreedy {
		t.Fatalf("DoLa output should be MORE surprising to the premature layer (%.3f vs %.3f)", preDoLa, preGreedy)
	}
	if bound := matGreedy + math.Log2(1/0.1); matDoLa > bound {
		t.Fatalf("plausibility guarantee violated: mature surprise %.3f > greedy %.3f + log2(1/α) = %.3f", matDoLa, matGreedy, bound)
	}
	// tightening α must tighten the mature bound in practice.
	dolaTight, err := nlp.DoLaDecode(m, prompt, maxNew, layers, 0.5, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	matTight := surprise(dolaTight, false)
	t.Logf("mature surprise at α=0.5: %.3f (α=0.1: %.3f)", matTight, matDoLa)
	if matTight >= matDoLa {
		t.Fatalf("tightening α should reduce mature surprise (α=0.5 %.3f vs α=0.1 %.3f)", matTight, matDoLa)
	}
}
