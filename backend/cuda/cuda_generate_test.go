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
	t.Logf("GENERATED: %q", strings.TrimSpace(renderPieces(genText)))
	t.Logf("FULL:\n%s", renderPieces(prompt+genText))
	if len(gen) == 0 {
		t.Fatal("generated nothing")
	}
}
