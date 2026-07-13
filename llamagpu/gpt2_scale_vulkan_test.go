//go:build vulkan && cgo

package llamagpu_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T544: the vulkan twin of §T543's at-scale pipeline test (never metal-only): the same
// synthetic 124M GPT-2-shaped checkpoint through GPT2FromHF → the VULKAN batched decoder.
// Same correctness probe (batched greedy ≡ full forward would need the slow analysis path
// again — instead pin batched-vulkan ≡ batched-metal-verified geometry contract and log
// throughput; the §T543 metal run already pinned batched ≡ analysis at this scale, and
// the vulkan recorder is §V3-cross-validated op by op).
func TestGPT2ScalePipelineVulkan(t *testing.T) {
	if testing.Short() {
		t.Skip("124M-parameter model; skipped in -short")
	}
	const vocab, ctxLen, d, layers, heads = 50257, 1024, 768, 12, 12
	g, err := nlp.GPT2FromHF(synthGPT2HF(vocab, ctxLen, d, layers), heads)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewGPTVulkan(g)
	if err != nil {
		t.Skip("vulkan decoder unavailable:", err)
	}
	defer dec.Release()

	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % vocab
	}
	seq, err := dec.Generate(prompt, 8, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(seq) != len(prompt)+8 {
		t.Fatalf("generated %d tokens, want 8", len(seq)-len(prompt))
	}
	for i, tok := range seq {
		if i < len(prompt) && tok != prompt[i] {
			t.Fatalf("prompt prefix violated at %d", i)
		}
		if tok < 0 || tok >= vocab {
			t.Fatalf("token %d out of vocab: %d", i, tok)
		}
	}
	const genN = 64
	start := time.Now()
	if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil {
		t.Fatal(err)
	}
	dt := time.Since(start).Seconds()
	t.Logf("GPT-2-124M-shape VULKAN batched decode: %d tokens in %.2fs = %.1f tok/s", genN, dt, float64(genN)/dt)
}
