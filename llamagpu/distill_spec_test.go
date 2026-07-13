//go:build darwin && cgo

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
)

// §T472: knowledge distillation closes the §T434 loop. §T434 measured draft-model speculative
// decoding with an INDEPENDENTLY trained draft (draft and target each trained on the corpus) at
// 81% acceptance. Speculative acceptance is exactly "how well does the draft match the TARGET's
// distribution" — which is what distillation optimizes directly. Here the same draft architecture
// is instead trained with nn.GKDLoss on the TEACHER'S LOGITS (β=0: forward KL, soft targets) over
// the same windows, and both drafts race on the same speculative task: the distilled draft must
// reach at least the independent draft's acceptance rate, and losslessness holds for both.
func TestDistilledDraftImprovesSpeculative(t *testing.T) {
	if testing.Short() {
		t.Skip("trains three models; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal not registered")
	}
	corpus := specCorpus(600)
	toks, _ := charTokens(corpus)
	vocab := 0
	for _, tk := range toks {
		if tk >= vocab {
			vocab = tk + 1
		}
	}
	tcfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}
	dcfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 48, Heads: 2, Layers: 1, Eps: 1e-5}
	const window = 128

	teacher := trainableGPT(t, tcfg, 1)
	tf, tl := trainCharGPT(t, teacher, be, toks, 240, window, 3e-3)
	t.Logf("teacher: CE %.3f → %.3f", tf, tl)

	// draft A: trained INDEPENDENTLY on the corpus (the §T434 setup).
	indep := trainableGPT(t, dcfg, 2)
	trainCharGPT(t, indep, be, toks, 240, window, 3e-3)

	// draft B: same architecture, DISTILLED from the teacher's logits over the same windows.
	distilled := trainableGPT(t, dcfg, 2) // same init as draft A — only the objective differs
	opt := nn.NewAdamW(distilled.Params(), 3e-3, 0.01)
	for step := range 240 {
		start := (step * 97) % (len(toks) - window - 1)
		win := toks[start : start+window]
		tctx := backend.NewContext().WithBackend(be)
		teacherLogits, err := teacher.Forward(tctx, win) // frozen teacher, outside the tape
		if err != nil {
			t.Fatal(err)
		}
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		studentLogits, err := distilled.Forward(ctx, win)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.GKDLoss(ctx, studentLogits, teacherLogits, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(distilled.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}

	// race: same speculative task, same target, both drafts.
	prompt := toks[:16]
	const maxNew = 96
	run := func(draft *nlp.GPT) (float64, []int) {
		target, err := llamagpu.NewGPT(teacher)
		if err != nil {
			t.Fatal(err)
		}
		defer target.Release()
		d, err := llamagpu.NewGPT(draft)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Release()
		out, stats, err := llamagpu.SpeculativeGenerate(target, d, prompt, maxNew, 4, nlp.Greedy(), 7)
		if err != nil {
			t.Fatal(err)
		}
		return stats.AcceptanceRate(), out
	}
	accIndep, outIndep := run(indep)
	accDist, outDist := run(distilled)

	// losslessness: both outputs equal the target's own greedy generation.
	plainDec, err := llamagpu.NewGPT(teacher)
	if err != nil {
		t.Fatal(err)
	}
	defer plainDec.Release()
	want, err := plainDec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if outIndep[i] != want[i] || outDist[i] != want[i] {
			t.Fatalf("losslessness broken at token %d", i)
		}
	}
	if accDist < accIndep {
		t.Fatalf("distilled draft acceptance %.0f%% < independent %.0f%% — distillation should match the teacher at least as well", 100*accDist, 100*accIndep)
	}
	t.Logf("speculative acceptance: independent draft %.0f%% vs GKD-distilled draft %.0f%% (same arch, same steps, same init)", 100*accIndep, 100*accDist)
}
