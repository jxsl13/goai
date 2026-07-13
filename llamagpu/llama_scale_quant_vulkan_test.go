//go:build vulkan && cgo

package llamagpu_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T546: the vulkan twin of §T545 (never metal-only): the same 124M-class Llama
// quantized and decoded through NewQuantVulkan; contract asserted, throughput logged.
func TestLlamaScaleQuantDecodeVulkan(t *testing.T) {
	if testing.Short() {
		t.Skip("124M-parameter model; skipped in -short")
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

	f32dec, err := llamagpu.NewVulkan(m)
	if err != nil {
		t.Skip("vulkan decoder unavailable:", err)
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
		qdec, err := llamagpu.NewQuantVulkan(qm)
		if err != nil {
			t.Fatal(err)
		}
		tps[q.name] = run(q.name, func() ([]int, error) { return qdec.Generate(prompt, genN, nlp.Greedy()) })
		qdec.Release()
		qm.Close()
	}
	t.Logf("124M-class Llama decode (VULKAN): f32 %.1f tok/s; Q8_0 %.1f; Q4_K %.1f", f32Tps, tps["Q8_0"], tps["Q4_K"])
}
