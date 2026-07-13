package nlp_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// a small SPM/UGM-style vocabulary as it appears in GGUF metadata: parallel
// tokenizer.ggml.tokens (strings) and tokenizer.ggml.scores (float32 log-probs).
func ggufTokenizerMeta(model string) map[string]any {
	tokens := []string{"<unk>", "▁", "a", "aa", "b", "▁a", "▁b"}
	scores := []float64{-20, -1, -1, -5, -1, -0.5, -0.5}
	toks := make([]any, len(tokens))
	scs := make([]any, len(tokens))
	for i := range tokens {
		toks[i] = tokens[i]
		scs[i] = float32(scores[i])
	}
	return map[string]any{
		"tokenizer.ggml.model":            model,
		"tokenizer.ggml.tokens":           toks,
		"tokenizer.ggml.scores":           scs,
		"tokenizer.ggml.unknown_token_id": uint32(0),
	}
}

// §R88: a Unigram tokenizer loads from GGUF tokenizer.ggml.* metadata and behaves as
// the Viterbi Unigram — "aa" segments as ▁+a+a (score −3) not ▁+aa (−6), proving the
// loaded scores drive the DP, and the <unk> id from metadata is honoured.
func TestUnigramFromGGUF(t *testing.T) {
	u, err := nlp.UnigramFromGGUF(ggufTokenizerMeta("t5"))
	if err != nil {
		t.Fatal(err)
	}
	got := u.Encode("aa")
	// "aa" → "▁aa" (dummy prefix): Viterbi maximizes score over the loaded pieces —
	// ▁a(−0.5)+a(−1) = −1.5 beats ▁(−1)+a+a (−3) and ▁+aa (−6).
	want := []int{5, 2} // ▁a, a
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("encode(aa) = %v, want %v (▁a+a is the max-score path)", got, want)
	}
	// an uncovered character falls back to the metadata <unk> id (0)
	if ids := u.Encode("z"); ids[len(ids)-1] != 0 {
		t.Errorf("uncovered char must map to unk id 0, got %v", ids)
	}
	if back := u.Decode(u.Encode("a b")); back != "a b" {
		t.Errorf("round-trip: %q", back)
	}
}

// §V15 / E2E: the tokenizer survives a full GGUF write→read cycle — build a File with
// tokenizer metadata, serialize it with the GGUF writer (T107), read it back, and load
// the tokenizer from the recovered metadata; it encodes identically to one built from
// the metadata directly. This exercises the whole gguf-writer + reader + loader chain.
func TestUnigramFromGGUFWriteReadRoundTrip(t *testing.T) {
	direct, err := nlp.UnigramFromGGUF(ggufTokenizerMeta("llama"))
	if err != nil {
		t.Fatal(err)
	}
	f := &gguf.File{Version: 3, Metadata: ggufTokenizerMeta("llama"), Tensors: map[string]*tensor.Tensor{}}
	var buf bytes.Buffer
	if err := gguf.Write(&buf, f); err != nil {
		t.Fatal(err)
	}
	back, err := gguf.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := nlp.UnigramFromGGUF(back.Metadata)
	if err != nil {
		t.Fatalf("load from round-tripped metadata: %v", err)
	}
	for _, s := range []string{"aa", "a b", "ab", "baa"} {
		if a, b := direct.Encode(s), loaded.Encode(s); fmt.Sprint(a) != fmt.Sprint(b) {
			t.Errorf("encode(%q): direct %v != gguf-round-trip %v", s, a, b)
		}
	}
}

// §R88: byte-level BPE model types and missing scores are rejected, not silently
// mis-loaded.
func TestUnigramFromGGUFErrors(t *testing.T) {
	if _, err := nlp.UnigramFromGGUF(ggufTokenizerMeta("gpt2")); err == nil {
		t.Error("gpt2 (byte BPE) must be rejected by the Unigram loader")
	}
	noScores := ggufTokenizerMeta("t5")
	delete(noScores, "tokenizer.ggml.scores")
	if _, err := nlp.UnigramFromGGUF(noScores); err == nil {
		t.Error("missing scores must error (Unigram needs scores)")
	}
	mismatch := ggufTokenizerMeta("t5")
	mismatch["tokenizer.ggml.scores"] = []any{float32(-1)} // wrong length
	if _, err := nlp.UnigramFromGGUF(mismatch); err == nil {
		t.Error("tokens/scores length mismatch must error")
	}
}

// A GGUF model file carries its own tokenizer; UnigramFromGGUF wires that embedded
// vocabulary to the Viterbi Unigram so a .gguf model tokenizes end-to-end.
func ExampleUnigramFromGGUF() {
	meta := map[string]any{
		"tokenizer.ggml.model":            "t5",
		"tokenizer.ggml.tokens":           []any{"<unk>", "▁the", "▁cat", "▁"},
		"tokenizer.ggml.scores":           []any{float32(-20), float32(-1), float32(-1), float32(-5)},
		"tokenizer.ggml.unknown_token_id": uint32(0),
	}
	u, _ := nlp.UnigramFromGGUF(meta)
	ids := u.Encode("the cat")
	fmt.Println(ids, u.Decode(ids))
	// Output: [1 2] the cat
}
