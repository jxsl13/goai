//go:build darwin && cgo

package llamagpu_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T543: GPT2FromHF at REAL SCALE — a 124M-parameter HF-shaped checkpoint (GPT-2 small
// geometry: 12 layers, d=768, vocab 50257, ctx 1024) converts, loads into the batched
// GPU decoder, and decodes. Asserts: geometry inferred correctly; the batched decoder's
// greedy tokens match the analysis-path full forward over a probe prefix (KV/bias/tied-
// head wiring correct at scale); logs prefill and decode throughput — the first
// real-model-size performance numbers of the pipeline. Random weights (see synthGPT2HF).
func TestGPT2ScalePipeline(t *testing.T) {
	if testing.Short() {
		t.Skip("124M-parameter model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	const vocab, ctxLen, d, layers, heads = 50257, 1024, 768, 12, 12
	g, err := nlp.GPT2FromHF(synthGPT2HF(vocab, ctxLen, d, layers), heads)
	if err != nil {
		t.Fatal(err)
	}
	cfg := g.Config
	if cfg.Vocab != vocab || cfg.Ctx != ctxLen || cfg.Dim != d || cfg.Layers != layers {
		t.Fatalf("geometry inferred wrong: %+v", cfg)
	}

	dec, err := llamagpu.NewGPT(g)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % vocab
	}

	// correctness probe: batched greedy continuation == analysis-path greedy (8 tokens).
	seq, err := dec.Generate(prompt, 8, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	ref := append([]int(nil), prompt...)
	for range 8 {
		logits, err := g.Forward(ctx, ref)
		if err != nil {
			t.Fatal(err)
		}
		best, row := 0, logits.Shape()[0]-1
		for v := 1; v < vocab; v++ {
			if logits.AtF64(row, v) > logits.AtF64(row, best) {
				best = v
			}
		}
		ref = append(ref, best)
	}
	for i := range ref {
		if seq[i] != ref[i] {
			t.Fatalf("batched decode diverges from full forward at %d: %d vs %d", i, seq[i], ref[i])
		}
	}

	// throughput (logged, not asserted — random weights, mechanics fire):
	const genN = 64
	start := time.Now()
	if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil {
		t.Fatal(err)
	}
	dt := time.Since(start).Seconds()
	t.Logf("GPT-2-124M-shape batched decode: %d tokens in %.2fs = %.1f tok/s (prompt %d)",
		genN, dt, float64(genN)/dt, len(prompt))
}
