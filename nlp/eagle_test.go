package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// eagleGreedyRef is plain greedy decoding of the base model via full Forward
// argmax — the exact reference EagleGenerate must reproduce token for token.
func eagleGreedyRef(t *testing.T, model *nlp.GPT, prompt []int, maxNew int) []int {
	t.Helper()
	ctx := backend.NewContext()
	out := append([]int(nil), prompt...)
	for len(out)-len(prompt) < maxNew && len(out) < model.Config.Ctx {
		lg, err := model.Forward(ctx, out)
		if err != nil {
			t.Fatal(err)
		}
		last := lg.Shape()[0] - 1
		best, bv := 0, math.Inf(-1)
		for v := 0; v < model.Config.Vocab; v++ {
			if x := lg.AtF64(last, v); x > bv {
				bv, best = x, v
			}
		}
		out = append(out, best)
	}
	return out
}

// trainEagleHead trains an EagleHead on the base model's own greedy rollouts
// from every start token (the EAGLE recipe: the head learns to predict the
// FROZEN base's feature walk; ForwardHidden runs outside the tape, the tape
// covers only EagleLoss). Returns the head and the first/last loss values.
func trainEagleHead(t *testing.T, model *nlp.GPT, seed uint64, steps int) (head *nlp.EagleHead, first, last float64) {
	t.Helper()
	cfg := model.Config
	head, err := nlp.NewEagleHead(tensor.F64, cfg.Dim, cfg.Vocab, seed)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	var seqs [][]int
	var hiddens []*tensor.Tensor
	for s0 := range cfg.Vocab {
		seq, err := model.Generate([]int{s0}, cfg.Ctx-1, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		h, err := model.ForwardHidden(ctx, seq) // frozen base: outside any tape
		if err != nil {
			t.Fatal(err)
		}
		seqs = append(seqs, seq)
		hiddens = append(hiddens, h)
	}
	opt := nn.NewAdamW(head.Params(), 0.02, 0.0)
	for step := range steps {
		n := step % len(seqs)
		tape := autograd.NewTape()
		loss, err := nlp.EagleLoss(tape.Context(), head, model, seqs[n], hiddens[n])
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(head.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	return head, first, last
}

// THE key EAGLE guarantee: EagleGenerate is LOSSLESS for ANY draft-head weights
// — even a random untrained head — because the base model verifies every drafted
// token and each round emits only base-greedy tokens. The output must equal
// plain greedy Forward-argmax decoding token for token, for several draft
// lengths and several random heads.
func TestEagleGenerateLossless(t *testing.T) {
	model, prompt := loadGPTModel(t)
	prompt = prompt[:2]
	maxNew := model.Config.Ctx - len(prompt)
	want := eagleGreedyRef(t, model, prompt, maxNew)

	for _, seed := range []uint64{3, 99} {
		head, err := nlp.NewEagleHead(tensor.F64, model.Config.Dim, model.Config.Vocab, seed)
		if err != nil {
			t.Fatal(err)
		}
		for _, draftLen := range []int{1, 2, 3, 5} {
			got, stats, err := nlp.EagleGenerate(model, head, prompt, maxNew, draftLen)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("seed %d draftLen %d: length %d vs greedy %d", seed, draftLen, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("seed %d draftLen %d token[%d]: eagle %d != greedy %d (losslessness broken)",
						seed, draftLen, i, got[i], want[i])
				}
			}
			if stats.Accepted > stats.Proposed {
				t.Fatalf("accepted %d > proposed %d", stats.Accepted, stats.Proposed)
			}
		}
	}
	t.Logf("EagleGenerate ≡ greedy decoding across 2 random heads × 4 draft lengths (%d tokens)", len(want))
}

// After training on the frozen base's own greedy rollouts the head drafts SOME
// correct tokens: total accepted speculative drafts > 0, i.e. the mean accepted
// length per round exceeds the trivial 1 (every round's first candidate is the
// base's exact greedy token; accepts count only the head's speculative drafts on
// top of it). The output must STILL be exactly greedy — training only buys speed.
func TestEagleAcceptanceAfterTraining(t *testing.T) {
	model, _ := loadGPTModel(t)
	cfg := model.Config
	head, _, _ := trainEagleHead(t, model, 7, 800)
	untrained, err := nlp.NewEagleHead(tensor.F64, cfg.Dim, cfg.Vocab, 7)
	if err != nil {
		t.Fatal(err)
	}

	var trainedStats, untrainedStats nlp.SpecStats
	for s0 := 0; s0 < cfg.Vocab; s0++ {
		prompt := []int{s0}
		maxNew := cfg.Ctx - 1
		want := eagleGreedyRef(t, model, prompt, maxNew)
		got, st, err := nlp.EagleGenerate(model, head, prompt, maxNew, 3)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("prompt %d token[%d]: trained-head eagle %d != greedy %d", s0, i, got[i], want[i])
			}
		}
		trainedStats.Proposed += st.Proposed
		trainedStats.Accepted += st.Accepted
		_, st, err = nlp.EagleGenerate(model, untrained, prompt, maxNew, 3)
		if err != nil {
			t.Fatal(err)
		}
		untrainedStats.Proposed += st.Proposed
		untrainedStats.Accepted += st.Accepted
	}
	if trainedStats.Proposed == 0 {
		t.Fatal("no drafts proposed")
	}
	if trainedStats.Accepted == 0 {
		t.Fatalf("trained head accepted nothing (%d proposed) — mean accepted length stuck at the trivial 1",
			trainedStats.Proposed)
	}
	t.Logf("acceptance: untrained %d/%d (%.0f%%) → trained %d/%d (%.0f%%)",
		untrainedStats.Accepted, untrainedStats.Proposed, 100*untrainedStats.AcceptanceRate(),
		trainedStats.Accepted, trainedStats.Proposed, 100*trainedStats.AcceptanceRate())
}

// The EAGLE loss (feature MSE + 0.1·token CE through the frozen base Head) at
// least halves under the library's own frozen-base training loop — gradients
// flow through Predict and both loss terms into the head parameters.
func TestEagleHeadTrains(t *testing.T) {
	model, _ := loadGPTModel(t)
	_, first, last := trainEagleHead(t, model, 5, 800)
	if last >= first*0.5 {
		t.Fatalf("EAGLE head did not train: loss %.4f → %.4f (want < half)", first, last)
	}
	t.Logf("EAGLE head trained on frozen base: loss %.4f → %.4f (800 steps)", first, last)
}

// Numeric gradcheck (F64): the analytic gradients of EagleLoss w.r.t. every
// EagleHead parameter entry match central finite differences (base frozen; only
// the head's parameters are checked — the base's tensors are constants here).
func TestEagleHeadGradcheck(t *testing.T) {
	model, tokens := loadGPTModel(t)
	head, err := nlp.NewEagleHead(tensor.F64, model.Config.Dim, model.Config.Vocab, 11, nlp.WithEagleFFNMult(2))
	if err != nil {
		t.Fatal(err)
	}
	hidden, err := model.ForwardHidden(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	tape := autograd.NewTape()
	loss, err := nlp.EagleLoss(tape.Context(), head, model, tokens, hidden)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	eval := func() float64 {
		l, err := nlp.EagleLoss(backend.NewContext(), head, model, tokens, hidden)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	const h = 1e-6
	var checked int
	var maxRel float64
	for pi, p := range head.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("param %d received no gradient", pi)
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			v := p.AtF64(idx...)
			p.SetF64(v+h, idx...)
			up := eval()
			p.SetF64(v-h, idx...)
			down := eval()
			p.SetF64(v, idx...)
			fd := (up - down) / (2 * h)
			diff := math.Abs(fd - g.AtF64(idx...))
			if rel := diff / math.Max(1, math.Abs(fd)); rel > maxRel {
				maxRel = rel
			}
			if diff > 1e-5*math.Max(1, math.Abs(fd)) {
				t.Fatalf("param %d entry %v: autograd %.10g vs fd %.10g", pi, idx, g.AtF64(idx...), fd)
			}
			checked++
		}
	}
	t.Logf("gradcheck passed: %d entries across %d params, max rel err %.3g", checked, len(head.Params()), maxRel)
}

// Constructor, Predict, EagleLoss and EagleGenerate reject bad geometry loudly.
func TestEagleShapesErrors(t *testing.T) {
	if _, err := nlp.NewEagleHead(tensor.F32, 0, 5, 1); err == nil {
		t.Fatal("dim=0 accepted")
	}
	if _, err := nlp.NewEagleHead(tensor.F32, 4, 0, 1); err == nil {
		t.Fatal("vocab=0 accepted")
	}
	if _, err := nlp.NewEagleHead(tensor.F32, 4, 5, 1, nlp.WithEagleFFNMult(0)); err == nil {
		t.Fatal("mult=0 accepted")
	}
	head, err := nlp.NewEagleHead(tensor.F64, 4, 6, 1)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	ok := tensor.New(tensor.F64, tensor.Shape{2, 4})
	if _, err := head.Predict(ctx, tensor.New(tensor.F64, tensor.Shape{2, 3}), ok); err == nil {
		t.Fatal("wrong feature width accepted")
	}
	if _, err := head.Predict(ctx, ok, tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Fatal("row mismatch accepted")
	}
	if out, err := head.Predict(ctx, ok, tensor.New(tensor.F64, tensor.Shape{2, 4})); err != nil {
		t.Fatal(err)
	} else if out.Shape()[0] != 2 || out.Shape()[1] != 4 {
		t.Fatalf("Predict shape %v, want (2, 4)", out.Shape())
	}

	model, tokens := loadGPTModel(t)
	if _, _, err := nlp.EagleGenerate(model, nil, tokens[:2], 4, 2); err == nil {
		t.Fatal("nil head accepted")
	}
	if _, _, err := nlp.EagleGenerate(model, head, tokens[:2], 4, 2); err == nil {
		t.Fatal("mismatched head geometry accepted")
	}
	big, err := nlp.NewEagleHead(tensor.F64, model.Config.Dim, model.Config.Vocab, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := nlp.EagleGenerate(model, big, nil, 4, 2); err == nil {
		t.Fatal("empty prompt accepted")
	}
	hidden, err := model.ForwardHidden(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.EagleLoss(ctx, big, model, tokens[:2], hidden); err == nil {
		t.Fatal("2-token sequence accepted")
	}
	if _, err := nlp.EagleLoss(ctx, big, model, tokens, tensor.New(tensor.F64, tensor.Shape{1, model.Config.Dim})); err == nil {
		t.Fatal("wrong hidden shape accepted")
	}
	if _, err := nlp.EagleLoss(ctx, head, model, tokens, hidden); err == nil {
		t.Fatal("mismatched head geometry accepted in EagleLoss")
	}
}

// EagleHead is EAGLE's one small autoregressive draft module: it predicts the
// base model's NEXT hidden feature from the current feature and the embedding of
// the token in between, reusing the frozen base's embedding table and LM head
// (contrast MedusaHeads: K parallel token heads; classic speculative decoding: a
// whole separate draft model). Here it maps a batch of (feature, embedding)
// pairs to the predicted next features.
func ExampleEagleHead() {
	head, err := nlp.NewEagleHead(tensor.F32, 4 /*dim*/, 6 /*vocab*/, 1 /*seed*/)
	if err != nil {
		panic(err)
	}
	feats := tensor.New(tensor.F32, tensor.Shape{3, 4})          // f_t rows, e.g. gpt.ForwardHidden
	embs := tensor.New(tensor.F32, tensor.Shape{3, 4})           // embeddings of tokens t+1
	next, err := head.Predict(backend.NewContext(), feats, embs) // row t ≈ f_{t+1}
	if err != nil {
		panic(err)
	}
	fmt.Println(len(head.Params()), next.Shape())
	// Output: 10 (3, 4)
}
