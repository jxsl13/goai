//go:build darwin && cgo

package llamagpu_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T485: NEFTune's documented integration path (Embed → nn.NEFTune → ForwardFromEmbed) verified
// on a real overfitting run, and its generalization claim measured. Setup engineered for the
// claim's regime: a SMALL training slice (overfit fodder) trained long, evaluated on held-out
// corpus text. Asserted: the noisy path trains (train CE drops sharply) and evaluation runs
// noise-free; the held-out comparison (plain vs α=5) is REPORTED — at toy scale regularization
// effects may or may not manifest (per the docs/inference.md discipline).
func TestNEFTuneOnTrainedGPT(t *testing.T) {
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
	trainSlice := toks[:1500] // small: induces overfitting
	held := toks[6000:6600]
	const steps, window = 300, 128

	train := func(alpha float64) (*nlp.GPT, float64) {
		m := trainableGPT(t, cfg, 1)
		rng := rand.New(rand.NewPCG(9, 0x9e77))
		opt := nn.NewAdamW(m.Params(), 3e-3, 0.01)
		targets := tensor.New(tensor.F32, tensor.Shape{window})
		var last float64
		for step := range steps {
			start := (step * 97) % (len(trainSlice) - window - 1)
			tokens := trainSlice[start : start+window]
			for i := range window {
				targets.SetF64(float64(trainSlice[start+1+i]), i)
			}
			tape := autograd.NewTapeOn(be)
			ctx := tape.Context()
			emb, err := m.Embed(ctx, tokens)
			if err != nil {
				t.Fatal(err)
			}
			if alpha > 0 { // the documented integration point: between embed and blocks
				if emb, err = nn.NEFTune(ctx, emb, alpha, rng); err != nil {
					t.Fatal(err)
				}
			}
			logits, err := m.ForwardFromEmbed(ctx, emb)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(ctx, logits, targets)
			if err != nil {
				t.Fatal(err)
			}
			last = loss.AtF64()
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
			if err := opt.Step(clipped); err != nil {
				t.Fatal(err)
			}
		}
		return m, last
	}
	// held-out CE (bits/token), noise-free (plain Forward — inference never sees noise).
	heldCE := func(m *nlp.GPT) float64 {
		ctx := backend.NewContext().WithBackend(be)
		var bits float64
		n := 0
		for start := 0; start+window+1 < len(held); start += window {
			tokens := held[start : start+window]
			logits, err := m.Forward(ctx, tokens)
			if err != nil {
				t.Fatal(err)
			}
			lp, err := nn.TokenLogProbs(ctx, logits, held[start+1:start+window+1])
			if err != nil {
				t.Fatal(err)
			}
			for i := range lp.Shape()[0] {
				bits += -lp.AtF64(i) / 0.6931471805599453
			}
			n += lp.Shape()[0]
		}
		return bits / float64(n)
	}

	plain, plainTrain := train(0)
	noisy, noisyTrain := train(5)
	plainHeld := heldCE(plain)
	noisyHeld := heldCE(noisy)
	t.Logf("train CE: plain %.3f | NEFTune(α=5) %.3f — held-out CE: plain %.3f | NEFTune %.3f (Δ %.3f)",
		plainTrain, noisyTrain, plainHeld, noisyHeld, plainHeld-noisyHeld)

	if plainTrain > 1.3 {
		t.Fatalf("plain baseline failed to train (CE %.3f)", plainTrain)
	}
	// a regularizer's train loss sits ABOVE the plain baseline by design — require learning,
	// not baseline-matching.
	if noisyTrain > plainTrain+0.5 {
		t.Fatalf("NEFTune path failed to train (train CE %.3f vs plain %.3f) — integration broken", noisyTrain, plainTrain)
	}
	if noisyHeld < plainHeld {
		t.Logf("FINDING: NEFTune improved held-out CE (%.3f vs %.3f) — the regularization claim shows at toy scale", noisyHeld, plainHeld)
	} else {
		t.Logf("FINDING: no held-out gain at toy scale (%.3f vs %.3f) — consistent with regularizers needing scale; the integration path itself is verified", noisyHeld, plainHeld)
	}
}
