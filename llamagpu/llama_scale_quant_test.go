//go:build darwin && cgo

package llamagpu_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T545: QUANTIZED decode at real model size — a 124M-class Llama (d=768, 12 layers,
// GQA 12/4, vocab 32000; deterministic random weights, mechanics+throughput fire like
// §T543) quantized with QuantizeLlama and decoded through the existing quant decoder.
// Asserts the generation contract per quant type and logs f32 vs Q8_0 vs Q4_K decode
// throughput — completing the at-scale table (weight memory: f32 ~500MB, Q8 ~130MB,
// Q4_K ~70MB).
func TestLlamaScaleQuantDecode(t *testing.T) {
	if testing.Short() {
		t.Skip("124M-parameter model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 32000, Ctx: 1024, Dim: 768, Heads: 12, KVHeads: 4, Layers: 12,
		Hidden: 2048, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*131 + 5) % cfg.Vocab
	}
	const genN = 64

	run := func(name string, gen func() ([]int, error)) float64 {
		seq, err := gen()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(seq) != len(prompt)+genN {
			t.Fatalf("%s: generated %d tokens, want %d", name, len(seq)-len(prompt), genN)
		}
		for i, tok := range seq {
			if i < len(prompt) && tok != prompt[i] {
				t.Fatalf("%s: prompt prefix violated at %d", name, i)
			}
			if tok < 0 || tok >= cfg.Vocab {
				t.Fatalf("%s: token out of vocab: %d", name, tok)
			}
		}
		start := time.Now()
		if _, err := gen(); err != nil {
			t.Fatal(err)
		}
		return float64(genN) / time.Since(start).Seconds()
	}

	f32dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	f32Tps := run("f32", func() ([]int, error) { return f32dec.Generate(prompt, genN, nlp.Greedy()) })
	f32dec.Release()

	tps := map[string]float64{}
	for _, q := range []struct {
		name string
		qt   gguf.QuantType
	}{{"Q8_0", gguf.Q8_0}, {"Q4_K", gguf.Q4_K}} {
		qm, err := nlp.QuantizeLlama(m, q.qt)
		if err != nil {
			t.Fatal(err)
		}
		qdec, err := llamagpu.NewQuant(qm)
		if err != nil {
			t.Fatal(err)
		}
		tps[q.name] = run(q.name, func() ([]int, error) { return qdec.Generate(prompt, genN, nlp.Greedy()) })
		qdec.Release()
		qm.Close()
	}
	t.Logf("124M-class Llama decode (metal): f32 %.1f tok/s; Q8_0 %.1f; Q4_K %.1f", f32Tps, tps["Q8_0"], tps["Q4_K"])
}
