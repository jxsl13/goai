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

	// Numerical reorder gate: the production decoder must retain the exact greedy continuation
	// against the analysis-path full forward for a long decode.
	const compareN = 256
	seq, err := dec.Generate(prompt, compareN, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	ref := append([]int(nil), prompt...)
	for range compareN {
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

// BenchmarkGPTDecodeStepMetal measures the public single-token Step boundary at GPT-2-small
// geometry. Model upload and the 16-token prefill are outside timing; positions advance through
// the KV cache and wrap below Ctx so fixed-count campaigns execute identical context lengths.
func BenchmarkGPTDecodeStepMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	const vocab, ctxLen, d, layers, heads = 50257, 1024, 768, 12, 12
	g, err := nlp.GPT2FromHF(synthGPT2HF(vocab, ctxLen, d, layers), heads)
	if err != nil {
		b.Fatal(err)
	}
	dec, err := llamagpu.NewGPT(g)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(dec.Release)

	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % vocab
	}
	if _, err := dec.StepNLast(prompt, 0); err != nil {
		b.Fatal(err)
	}

	pos := len(prompt)
	// Compile pipelines and overwrite the same first decode row outside timing.
	if _, err := dec.Step((pos*97+11)%vocab, pos); err != nil {
		b.Fatal(err)
	}
	pos++
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if pos >= ctxLen {
			pos = len(prompt) + 1
		}
		token := (pos*97 + 11) % vocab
		if _, err := dec.Step(token, pos); err != nil {
			b.Fatal(err)
		}
		pos++
	}
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

// BenchmarkGPTPrefillLastMetal measures the generation prefill boundary at GPT-2-small geometry:
// StepNLast populates the KV cache for every prompt row but projects and downloads only the last
// row's logits. Repeated iterations overwrite the same cache rows and execute identical work.
func BenchmarkGPTPrefillLastMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	const vocab, ctxLen, d, layers, heads = 50257, 1024, 768, 12, 12
	g, err := nlp.GPT2FromHF(synthGPT2HF(vocab, ctxLen, d, layers), heads)
	if err != nil {
		b.Fatal(err)
	}
	dec, err := llamagpu.NewGPT(g)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(dec.Release)
	for _, shape := range []struct {
		name string
		rows int
	}{
		{name: "pp16", rows: 16},
		{name: "pp64", rows: 64},
		{name: "pp256", rows: 256},
	} {
		prompt := make([]int, shape.rows)
		for i := range prompt {
			prompt[i] = (i*97 + 11) % vocab
		}
		b.Run(shape.name, func(b *testing.B) {
			if _, err := dec.StepNLast(prompt, 0); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := dec.StepNLast(prompt, 0); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(shape.rows*b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

// BenchmarkGPTDecodeStepIntoMetal measures allocation-free GPT decode with caller-owned logits.
func BenchmarkGPTDecodeStepIntoMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	const vocab, ctxLen, d, layers, heads = 50257, 1024, 768, 12, 12
	g, err := nlp.GPT2FromHF(synthGPT2HF(vocab, ctxLen, d, layers), heads)
	if err != nil {
		b.Fatal(err)
	}
	dec, err := llamagpu.NewGPT(g)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(dec.Release)
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % vocab
	}
	if _, err := dec.StepNLast(prompt, 0); err != nil {
		b.Fatal(err)
	}
	out := make([]float32, vocab)
	pos := len(prompt)
	if err := dec.StepInto((pos*97+11)%vocab, pos, out); err != nil {
		b.Fatal(err)
	}
	pos++
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if pos >= ctxLen {
			pos = len(prompt) + 1
		}
		if err := dec.StepInto((pos*97+11)%vocab, pos, out); err != nil {
			b.Fatal(err)
		}
		pos++
	}
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

// BenchmarkGPTPrefillLastIntoMetal measures allocation-free 16-token GPT prefill.
func BenchmarkGPTPrefillLastIntoMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	const vocab, ctxLen, d, layers, heads = 50257, 1024, 768, 12, 12
	g, err := nlp.GPT2FromHF(synthGPT2HF(vocab, ctxLen, d, layers), heads)
	if err != nil {
		b.Fatal(err)
	}
	dec, err := llamagpu.NewGPT(g)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(dec.Release)
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % vocab
	}
	out := make([]float32, vocab)
	if err := dec.StepNLastInto(prompt, 0, out); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := dec.StepNLastInto(prompt, 0, out); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(prompt)*b.N)/b.Elapsed().Seconds(), "tok/s")
}
