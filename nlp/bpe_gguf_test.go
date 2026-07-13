package nlp_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (§R90): BPE merges by RANK (lowest merge-list index first), NOT left to
// right. For "abc" with merges "b c"(rank 0) before "a b"(rank 1), the b+c merge fires
// first → symbols [a, bc], never [ab, c] — proving rank-ordered merging.
func TestBPEMergeByRank(t *testing.T) {
	tok, err := nlp.NewBPE(
		[]string{"a", "b", "c", "bc"}, // ids 0,1,2,3 — note "ab" is deliberately absent
		[]string{"b c", "a b"},        // "b c" has the lower (earlier) rank
	)
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Encode("abc")
	want := []int{0, 3} // a, bc  (rank-0 merge b+c wins over left-first a+b)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("encode(abc) = %v, want %v (b+c merges before a+b)", got, want)
	}
}

// gpt2 GGUF tokenizer metadata: byte-mapped tokens + space-separated merges.
func bpeGGUFMeta(model string) map[string]any {
	// ids 0..8; "hell" and the byte-mapped space "Ġ" (U+0120) are appended so every
	// final symbol the merges produce resolves in the vocabulary.
	tokens := []string{"h", "e", "l", "o", "he", "ll", "hello", "hell", "Ġ"}
	merges := []string{"h e", "l l", "he ll", "hell o"}
	toks := make([]any, len(tokens))
	for i := range tokens {
		toks[i] = tokens[i]
	}
	mrg := make([]any, len(merges))
	for i := range merges {
		mrg[i] = merges[i]
	}
	return map[string]any{
		"tokenizer.ggml.model":  model,
		"tokenizer.ggml.tokens": toks,
		"tokenizer.ggml.merges": mrg,
	}
}

// §R88/§R90: a byte-level BPE tokenizer loads from GGUF gpt2/bpe metadata and applies
// the merges — "hello" builds up he→hell→hello (id 6).
func TestBPEFromGGUF(t *testing.T) {
	tok, err := nlp.BPEFromGGUF(bpeGGUFMeta("gpt2"))
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("hello"); fmt.Sprint(got) != fmt.Sprint([]int{6}) {
		t.Errorf("encode(hello) = %v, want [6] (he+ll+o merged)", got)
	}
	// SentencePiece model types are routed to the Unigram loader, not this one
	if _, err := nlp.BPEFromGGUF(bpeGGUFMeta("t5")); err == nil {
		t.Error("t5 (Unigram) must be rejected by the BPE loader")
	}
}

// §V15 / E2E: the BPE tokenizer survives a full GGUF write→read cycle — build a File
// with gpt2 tokenizer metadata, serialize it (T107), read it back, and load the
// tokenizer; it encodes identically to one built from the metadata directly.
func TestBPEFromGGUFWriteReadRoundTrip(t *testing.T) {
	direct, err := nlp.BPEFromGGUF(bpeGGUFMeta("gpt2"))
	if err != nil {
		t.Fatal(err)
	}
	f := &gguf.File{Version: 3, Metadata: bpeGGUFMeta("gpt2"), Tensors: map[string]*tensor.Tensor{}}
	var buf bytes.Buffer
	if err := gguf.Write(&buf, f); err != nil {
		t.Fatal(err)
	}
	back, err := gguf.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := nlp.BPEFromGGUF(back.Metadata)
	if err != nil {
		t.Fatalf("load from round-tripped metadata: %v", err)
	}
	for _, s := range []string{"hello", "hell", "he", "hello hello"} {
		if a, b := direct.Encode(s), loaded.Encode(s); fmt.Sprint(a) != fmt.Sprint(b) {
			t.Errorf("encode(%q): direct %v != gguf-round-trip %v", s, a, b)
		}
		if back := loaded.Decode(loaded.Encode(s)); back != s {
			t.Errorf("round-trip decode(%q) = %q", s, back)
		}
	}
}

// A byte-level BPE tokenizer merges characters into subwords by rank: with the merge
// "h i", the two letters of "hi" combine into the single token "hi", and decoding
// restores the text.
func ExampleBPETokenizer() {
	tok, _ := nlp.NewBPE([]string{"h", "i", "hi"}, []string{"h i"})
	ids := tok.Encode("hi")
	fmt.Println(ids, tok.Decode(ids))
	// Output: [2] hi
}
