//go:build darwin && cgo

package llamagpu_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T446: with an unreachable acceptance threshold every head proposal is rejected, so batched
// Medusa must reproduce the batched decoder's own greedy Generate token for token — whatever
// the (untrained) heads propose, and despite the wasted verify rows (position-addressed
// overwrite is the rollback).
func TestMedusaGenerateGPTAllRejectIsGreedy(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.GPTConfig{Vocab: 48, Ctx: 64, Dim: 64, Heads: 4, Layers: 2, Eps: 1e-5}
	m := randGPT(t, cfg)
	heads, err := nlp.NewMedusaHeads(3, cfg.Dim, cfg.Vocab, 9)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	prompt := []int{5, 12, 3}
	const maxNew = 24
	got, stats, err := llamagpu.MedusaGenerate(dec, heads, prompt, maxNew, 1.0, 1e9)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Release()
	want, err := ref.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: medusa %d != plain %d", i, got[i], want[i])
		}
	}
	if stats.Accepted != 0 || stats.Proposed == 0 {
		t.Fatalf("ε=1: accepted %d / proposed %d, want 0 / >0", stats.Accepted, stats.Proposed)
	}
}

// §T446 measurement: on a base trained in-repo (§T434 infra) with heads trained on the base's
// OWN batched-greedy rollouts, batched Medusa reaches real acceptance — and because the heads
// draft for free (host projections, no draft decoder steps), a round costs ~2 steps for up to
// K+1 tokens. Reports acceptance and the measured A/B throughput.
func TestMedusaGenerateGPTTrainedThroughput(t *testing.T) {
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
	t.Logf("base trained: CE %.3f → %.3f", tf, tl)

	// rollouts from the BASE itself (batched decoder for speed), hidden via ForwardHidden.
	roll, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	const K = 3
	heads, err := nlp.NewMedusaHeads(K, cfg.Dim, cfg.Vocab, 11)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext().WithBackend(be)
	var seqs [][]int
	var hiddens []*tensor.Tensor
	for r := range 6 {
		prompt := toks[r*31 : r*31+16]
		seq, err := roll.Generate(prompt, 120, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		h, err := m.ForwardHidden(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		seqs = append(seqs, seq)
		hiddens = append(hiddens, h)
	}
	roll.Release()

	opt := nn.NewAdamW(heads.Params(), 0.02, 0.0)
	var hFirst, hLast float64
	for step := range 300 {
		n := step % len(seqs)
		s := seqs[n]
		tape := autograd.NewTape()
		tctx := tape.Context()
		logits, err := heads.Logits(tctx, hiddens[n])
		if err != nil {
			t.Fatal(err)
		}
		var total *tensor.Tensor
		for k := range K {
			rows := len(s) - 2 - k
			targets := tensor.New(tensor.F32, tensor.Shape{rows})
			for i := range rows {
				targets.SetF64(float64(s[i+2+k]), i)
			}
			sub, err := backend.Execute(tctx, backend.OpSlice, []*tensor.Tensor{logits[k]},
				backend.SliceAttrs{Axis: 0, Start: 0, End: rows})
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(tctx, sub[0], targets)
			if err != nil {
				t.Fatal(err)
			}
			if total == nil {
				total = loss
			} else {
				o, err := backend.Execute(tctx, backend.OpAdd, []*tensor.Tensor{total, loss}, nil)
				if err != nil {
					t.Fatal(err)
				}
				total = o[0]
			}
		}
		lv := total.AtF64()
		if step == 0 {
			hFirst = lv
		}
		hLast = lv
		if err := tape.Backward(total); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(heads.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("heads trained on base rollouts: CE %.3f → %.3f", hFirst, hLast)

	prompt := toks[:16]
	const maxNew = 96
	medusaDec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer medusaDec.Release()
	out, stats, err := llamagpu.MedusaGenerate(medusaDec, heads, prompt, maxNew, -1, -1)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("length %d, want %d", len(out), len(prompt)+maxNew)
	}
	if stats.Proposed == 0 || stats.Accepted == 0 {
		t.Fatalf("trained heads: accepted %d / proposed %d — accept path never fired", stats.Accepted, stats.Proposed)
	}

	median := func(f func() error) float64 {
		var ms [3]float64
		for i := range ms {
			s := time.Now()
			if err := f(); err != nil {
				t.Fatal(err)
			}
			ms[i] = time.Since(s).Seconds() * 1e3
		}
		if ms[0] > ms[1] {
			ms[0], ms[1] = ms[1], ms[0]
		}
		if ms[1] > ms[2] {
			ms[1], ms[2] = ms[2], ms[1]
		}
		if ms[0] > ms[1] {
			ms[0], ms[1] = ms[1], ms[0]
		}
		return ms[1]
	}
	tMed := median(func() error {
		_, _, err := llamagpu.MedusaGenerate(medusaDec, heads, prompt, maxNew, -1, -1)
		return err
	})
	plainDec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer plainDec.Release()
	tPlain := median(func() error {
		_, err := plainDec.Generate(prompt, maxNew, nlp.Greedy())
		return err
	})
	t.Logf("acceptance %.0f%% (%d/%d); plain %.0f ms (%.0f tok/s) vs medusa %.0f ms (%.0f tok/s) → speedup %.2fx",
		100*stats.AcceptanceRate(), stats.Accepted, stats.Proposed,
		tPlain, float64(maxNew)/tPlain*1e3, tMed, float64(maxNew)/tMed*1e3, tPlain/tMed)
}

// §T447: the same MedusaGenerate loop drives the LLAMA batched decoder through HiddenStepper —
// the ε=1 collapse onto plain batched greedy Generate must hold there too.
func TestMedusaGenerateLlamaAllRejectIsGreedy(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 64, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	heads, err := nlp.NewMedusaHeads(3, cfg.Dim, cfg.Vocab, 9)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	prompt := []int{5, 12, 3}
	const maxNew = 24
	got, stats, err := llamagpu.MedusaGenerate(dec, heads, prompt, maxNew, 1.0, 1e9)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Release()
	want, err := ref.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: medusa %d != plain %d", i, got[i], want[i])
		}
	}
	if stats.Accepted != 0 || stats.Proposed == 0 {
		t.Fatalf("ε=1: accepted %d / proposed %d", stats.Accepted, stats.Proposed)
	}
}

// §T447: Llama.ForwardHidden times the output projection reproduces Forward bit-identically —
// the head attachment point for Llama Medusa training.
func TestLlamaForwardHiddenTimesOutIsForward(t *testing.T) {
	cfg := nlp.LlamaConfig{
		Vocab: 32, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2,
		Hidden: 88, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	tokens := []int{1, 2, 3, 4, 5}
	want, err := m.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	hidden, err := m.ForwardHidden(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	got, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{hidden, m.Out}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !got[0].Shape().Equal(want.Shape()) {
		t.Fatalf("shape %v vs %v", got[0].Shape(), want.Shape())
	}
	for i := range want.Numel() {
		idx := tensor.Unravel(i, want.Shape())
		if a, b := got[0].AtF64(idx...), want.AtF64(idx...); a != b {
			t.Fatalf("[%v]: %.17g != %.17g", idx, a, b)
		}
	}
}

// §T455: with a vanishing threshold every proposal is accepted — the lead-window bookkeeping
// (positions, budget cut mid-round, stats) must stay exact and deterministic.
func TestMedusaGenerateAcceptAllBookkeeping(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.GPTConfig{Vocab: 48, Ctx: 64, Dim: 64, Heads: 4, Layers: 2, Eps: 1e-5}
	m := randGPT(t, cfg)
	heads, err := nlp.NewMedusaHeads(3, cfg.Dim, cfg.Vocab, 9)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	prompt := []int{5, 12, 3}
	const maxNew = 26 // not a multiple of K+1: exercises the mid-round budget cut
	out1, stats, err := llamagpu.MedusaGenerate(dec, heads, prompt, maxNew, 1e-12, 1e-12)
	if err != nil {
		t.Fatal(err)
	}
	if len(out1) != len(prompt)+maxNew {
		t.Fatalf("length %d, want %d", len(out1), len(prompt)+maxNew)
	}
	if stats.Accepted != stats.Proposed || stats.Proposed == 0 {
		t.Fatalf("accept-all: %d/%d", stats.Accepted, stats.Proposed)
	}
	for _, tok := range out1 {
		if tok < 0 || tok >= cfg.Vocab {
			t.Fatalf("invalid token %d", tok)
		}
	}
	out2, _, err := llamagpu.MedusaGenerate(dec, heads, prompt, maxNew, 1e-12, 1e-12)
	if err != nil {
		t.Fatal(err)
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatal("MedusaGenerate must be deterministic")
		}
	}
}
