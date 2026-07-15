//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"strings"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// End-to-end validation of the UNIFIED public API on a REAL model: load TinyLlama-1.1B (f32) + its
// SentencePiece tokenizer, drive the whole decode through llamagpu.NewCUDA (the backend-agnostic
// Decoder over the cuda.Recorder — every op on the GPU), and check it writes the correct text.
//
// This exercises the full f32 stack (embed, RMSNorm, fused-QKV RoPEPair, GQA attention, SwiGLU,
// residuals, vocab head) at real 2048-dim / 22-layer scale through the unified path — the earlier
// adapter tests used synthetic small models; this proves the public API is correct on a real model
// with a real tokenizer, the way a user calls it.
func TestCUDAUnifiedRealModelGenerate(t *testing.T) {
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := nlp.UnigramFromGGUF(f.Metadata)
	if err != nil {
		t.Fatal(err)
	}

	dec, err := llamagpu.NewCUDA(model)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	const maxGen = 24
	prompt := "The capital of France is"
	ids := append([]int{1}, tok.Encode(prompt)...) // 1 = BOS

	out, err := dec.Generate(ids, maxGen, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	gen := out[len(ids):]
	genText := renderPieces(tok.Decode(gen))
	t.Logf("PROMPT:    %q", prompt)
	t.Logf("UNIFIED llamagpu.NewCUDA (f32) greedy: %q", strings.TrimSpace(genText))

	if len(gen) == 0 {
		t.Fatal("generated nothing")
	}
	// The model is confident here: greedy must produce "Paris" — a real-text correctness gate on
	// the entire unified GPU stack (matches the bespoke demo, PR#57).
	if !strings.Contains(strings.ToLower(genText), "paris") {
		t.Fatalf("unified real-model greedy did not produce 'Paris': %q", strings.TrimSpace(genText))
	}
	t.Log("unified public API writes correct real text on TinyLlama-1.1B ✓")
}
