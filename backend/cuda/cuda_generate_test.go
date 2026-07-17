//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// End-to-end demo: load TinyLlama + its tokenizer, encode a prompt, run the
// optimized Q8 fixed-buffer GPU decode, and DETOKENIZE the generated ids back to
// text — the whole pipeline (tokenizer → resident Q8 model → device decode →
// argmax → tokenizer) producing real, readable output.
// renderPieces turns SentencePiece byte-fallback tokens (<0xXX>) into their bytes
// so the demo output is readable (a display nicety, not model logic).
var byteTok = regexp.MustCompile(`<0x([0-9A-Fa-f]{2})>`)

func renderPieces(s string) string {
	return byteTok.ReplaceAllStringFunc(s, func(m string) string {
		b, _ := strconv.ParseUint(m[3:5], 16, 8)
		return string([]byte{byte(b)})
	})
}

func TestCUDATinyLlamaGenerate(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
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

	const maxGen = 32
	prompt := "The capital of France is"
	ids := append([]int{1}, tok.Encode(prompt)...) // 1 = BOS

	gd := buildQ8GraphDecoder(t, model, len(ids)+maxGen+2)
	defer gd.free()

	// prefill the prompt token-by-token
	var last int
	for i, id := range ids {
		gd.step(t, int32(id), i)
		last = gd.logits.Argmax()
	}
	// greedy generate until EOS (2) or maxGen
	gen := make([]int, 0, maxGen)
	pos := len(ids)
	for n := 0; n < maxGen; n++ {
		if last == 2 { // EOS
			break
		}
		gen = append(gen, last)
		gd.step(t, int32(last), pos)
		last = gd.logits.Argmax()
		pos++
	}

	genText := tok.Decode(gen)
	t.Logf("PROMPT:    %q", prompt)
	t.Logf("GENERATED (greedy): %q", strings.TrimSpace(renderPieces(genText)))
	if len(gen) == 0 {
		t.Fatal("generated nothing")
	}
}

// Sampled generation on the GPU: reuse the full nlp.Sampler (temperature + top-k +
// repeat penalty) over the device logits — greedy on a 1.1B model degenerates
// (see above), sampling produces varied, non-repeating text. The decode runs on
// the optimized Q8 GPU path; only the per-token draw is host-side.
func TestCUDATinyLlamaGenerateSampled(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
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

	const maxGen = 40
	prompt := "The capital of France is"
	ids := append([]int{1}, tok.Encode(prompt)...)
	gd := buildQ8GraphDecoder(t, model, len(ids)+maxGen+2)
	defer gd.free()

	sampler := nlp.NewSampler(1234,
		nlp.WithTemperature(0.8), nlp.WithTopK(40), nlp.WithTopP(0.95))
	sampler.RepeatPenalty = 1.3 // the fix for 1.1B greedy degeneration

	// logitsF64 pulls the current [1,vocab] device logits to host as []float64.
	logitsF64 := func() []float64 {
		h, err := gd.logits.ToHost()
		mustTB(t, err)
		v := h.Shape()[1]
		out := make([]float64, v)
		for j := 0; j < v; j++ {
			out[j] = h.AtF64(0, j)
		}
		return out
	}

	for i, id := range ids {
		gd.step(t, int32(id), i)
	}
	history := append([]int{}, ids...)
	gen := make([]int, 0, maxGen)
	pos := len(ids)
	for n := 0; n < maxGen; n++ {
		tokID := sampler.SampleWithHistory(logitsF64(), history)
		if tokID == 2 { // EOS
			break
		}
		gen = append(gen, tokID)
		history = append(history, tokID)
		gd.step(t, int32(tokID), pos)
		pos++
	}

	t.Logf("PROMPT:    %q", prompt)
	t.Logf("GENERATED (sampled, T=0.8 topK=40 topP=0.95 rep=1.3): %q", strings.TrimSpace(renderPieces(tok.Decode(gen))))
	t.Logf("FULL:\n%s", renderPieces(prompt+tok.Decode(gen)))
	if len(gen) == 0 {
		t.Fatal("generated nothing")
	}
}
