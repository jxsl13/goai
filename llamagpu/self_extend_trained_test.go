//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// teacherForcedCE returns the mean cross-entropy of logits rows [from, to) against
// the next-token targets from toks.
func teacherForcedCE(t *testing.T, logits *tensor.Tensor, toks []int, from, to int) float64 {
	t.Helper()
	var ce float64
	for p := from; p < to; p++ {
		var m float64 = math.Inf(-1)
		vocab := logits.Shape()[1]
		for v := 0; v < vocab; v++ {
			if l := logits.AtF64(p, v); l > m {
				m = l
			}
		}
		var sum float64
		for v := 0; v < vocab; v++ {
			sum += math.Exp(logits.AtF64(p, v) - m)
		}
		ce -= logits.AtF64(p, toks[p+1]) - m - math.Log(sum)
	}
	return ce / float64(to-from)
}

// §T513 e2e: Self-Extend (arXiv:2401.01325) on a REAL trained model. A char-Llama is
// trained ONLY on windows of length 32 (relative positions 0..31), then teacher-forced
// over a 128-token held-out stretch — 4× its training length. Plain forward meets
// relative positions it never saw (the paper's premise: RoPE extrapolation degrades);
// SelfExtendForward with window 8 / group 8 compresses every distant pair back into
// the trained range (max effective rel pos ⌊127/8⌋+8−1 = 22 < 32) with NO fine-tuning.
// Claim under test: Self-Extend's CE on the far half (positions 64..127, all beyond
// training length) beats the plain forward's.
func TestSelfExtendBeyondTrainingLength(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model on cpu; skipped in -short")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.LlamaConfig{
		Vocab: vocab, Ctx: 160, Dim: 64, Heads: 4, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	const trainWindow = 32
	first, last := trainCharLlama(t, m, toks, 400, trainWindow, 3e-3)
	t.Logf("trained on windows of %d: CE %.3f → %.3f", trainWindow, first, last)

	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend not registered")
	}
	ctx := backend.NewContext().WithBackend(be)

	const evalLen, farFrom = 128, 64
	const window, group = 8, 8
	var plainFar, seFar, seNear float64
	const runs = 3
	for r := range runs {
		start := 4000 + r*137 // held-out stretches (training sampled step*97 mod …)
		seq := toks[start : start+evalLen+1]
		plain, err := m.Forward(ctx, seq[:evalLen])
		if err != nil {
			t.Fatal(err)
		}
		se, err := m.SelfExtendForward(ctx, seq[:evalLen], window, group)
		if err != nil {
			t.Fatal(err)
		}
		plainFar += teacherForcedCE(t, plain, seq, farFrom, evalLen) / runs
		seFar += teacherForcedCE(t, se, seq, farFrom, evalLen) / runs
		seNear += teacherForcedCE(t, se, seq, 1, trainWindow) / runs
	}
	t.Logf("far half (pos %d..%d, 2–4× training length): plain CE %.3f vs self-extend CE %.3f; self-extend near CE %.3f",
		farFrom, evalLen-1, plainFar, seFar, seNear)
	if seFar >= plainFar {
		t.Fatalf("self-extend CE %.3f did not beat plain %.3f beyond training length", seFar, plainFar)
	}
}

// §T541: SelfExtendGenerate — GENERATION beyond training length. The same trained
// char-Llama generates 96 tokens past a 32-token prompt (total 4× its training
// window); quality = teacher-forced surprise of the model on its OWN generated
// text, far half vs the plain greedy generation's far half. Plain RoPE generation
// degenerates out there (§T527's premise); Self-Extend must stay coherent enough
// to beat it. Contract: prompt prefix, in-vocab, exact length.
func TestSelfExtendGenerateBeyondTrainingLength(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model on cpu; skipped in -short")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.LlamaConfig{
		Vocab: vocab, Ctx: 160, Dim: 64, Heads: 4, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	const trainWindow = 32
	_, _ = trainCharLlama(t, m, toks, 400, trainWindow, 3e-3)

	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend not registered")
	}
	ctx := backend.NewContext().WithBackend(be)
	prompt := toks[:32]
	const maxNew, window, group = 96, 8, 8

	se, err := m.SelfExtendGenerate(ctx, prompt, maxNew, window, group)
	if err != nil {
		t.Fatal(err)
	}
	if len(se) != len(prompt)+maxNew {
		t.Fatalf("generated %d tokens, want %d", len(se)-len(prompt), maxNew)
	}
	for i, tok := range se {
		if i < len(prompt) && tok != prompt[i] {
			t.Fatalf("output does not start with the prompt at %d", i)
		}
		if tok < 0 || tok >= vocab {
			t.Fatalf("token %d out of vocab: %d", i, tok)
		}
	}
	// plain greedy generation for the same budget (group=1 ⇒ ordinary attention).
	plain, err := m.SelfExtendGenerate(ctx, prompt, maxNew, window, 1)
	if err != nil {
		t.Fatal(err)
	}
	// quality: windowed teacher-forced surprise of the model on each generated far half
	// (positions beyond 2× the training window), scored with SHORT in-window forwards
	// (the model's own competence range — the §T476 protocol).
	surprise := func(seq []int) float64 {
		var s float64
		n := 0
		for p := 64; p < len(seq)-1; p++ {
			lo := p - trainWindow + 1
			logits, err := m.Forward(ctx, seq[lo:p+1])
			if err != nil {
				t.Fatal(err)
			}
			s += teacherForcedCE(t, logits, seq[lo:], logits.Shape()[0]-1, logits.Shape()[0])
			n++
		}
		return s / float64(n)
	}
	seS, plainS := surprise(se), surprise(plain)
	t.Logf("far-half windowed surprise: self-extend %.3f vs plain greedy %.3f", seS, plainS)
	if seS >= plainS {
		t.Fatalf("self-extend generation (%.3f) did not beat plain (%.3f) beyond training length", seS, plainS)
	}
}
