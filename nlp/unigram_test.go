package nlp_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §V16 tier-1: encode is the 1-best VITERBI, not greedy longest-match. With pieces
// "a"(−1) and "aa"(−5), greedy-longest tiles "aa" as one token (total −5) but the
// max-probability segmentation is "a"+"a" (total −2). Encode must return the Viterbi
// choice ["a","a"].
func TestUnigramViterbiNotGreedy(t *testing.T) {
	u, err := nlp.NewUnigram([]nlp.UnigramPiece{
		{"▁", -1}, {"a", -1}, {"aa", -5},
	}, nlp.WithUnigramDummyPrefix(false))
	if err != nil {
		t.Fatal(err)
	}
	got := u.Encode("aa")
	want := []int{1, 1} // "a","a", NOT "aa"(id 2)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("Viterbi encode = %v, want %v (a+a beats aa)", got, want)
	}
}

// §V16 tier-1: when a longer piece IS the higher-probability tiling the Viterbi picks
// it — "aa"(−1) now beats "a"+"a"(−4).
func TestUnigramViterbiPicksLongWhenBetter(t *testing.T) {
	u, _ := nlp.NewUnigram([]nlp.UnigramPiece{
		{"a", -2}, {"aa", -1},
	}, nlp.WithUnigramDummyPrefix(false))
	if got := u.Encode("aa"); fmt.Sprint(got) != fmt.Sprint([]int{1}) {
		t.Errorf("encode = %v, want [1] (aa beats a+a)", got)
	}
}

// §V16 tier-1 / §V15: with full single-character coverage the tokenizer round-trips
// any input, including spaces (▁ escaping + add_dummy_prefix must invert exactly).
func TestUnigramRoundTrip(t *testing.T) {
	u, _ := nlp.NewUnigram([]nlp.UnigramPiece{
		{"<unk>", -20},
		{"▁", -3}, {"h", -2}, {"e", -2}, {"l", -2}, {"o", -2}, {"w", -2}, {"r", -2}, {"d", -2},
		{"▁h", -1}, {"he", -1}, {"ll", -1}, {"lo", -1}, {"▁w", -1}, {"or", -1}, {"ld", -1},
		{"hello", -0.5}, {"▁world", -0.5},
	})
	for _, s := range []string{"hello", "hello world", "world hello", "hollow"} {
		if back := u.Decode(u.Encode(s)); back != s {
			t.Errorf("round-trip %q → %q", s, back)
		}
	}
}

// §V16 tier-1: a character no piece covers becomes a single <unk> token at score
// min−kUnkPenalty, and the DP still returns a valid path.
func TestUnigramUnknown(t *testing.T) {
	u, _ := nlp.NewUnigram([]nlp.UnigramPiece{
		{"<unk>", -20}, {"▁", -1}, {"a", -1},
	}, nlp.WithUnigramDummyPrefix(false))
	got := u.Encode("axa") // x uncovered → <unk> (id 0)
	want := []int{2, 0, 2} // a, <unk>, a
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("encode(axa) = %v, want %v", got, want)
	}
}

// The Unigram tokenizer segments by the most-probable path over its scored vocabulary
// and detokenizes losslessly. Here "▁hello"+"▁world" is the best segmentation and
// decode restores the spacing.
func ExampleUnigram() {
	u, _ := nlp.NewUnigram([]nlp.UnigramPiece{
		{"<unk>", -20}, {"▁hello", -1}, {"▁world", -1}, {"▁", -5},
	})
	ids := u.Encode("hello world")
	fmt.Println(ids, u.Decode(ids))
	// Output: [1 2] hello world
}

// deepseekVocab builds a Unigram vocabulary that can spell a DeepSeek-style control
// token whose NAME contains U+2581, plus the pieces to spell ordinary text.
func deepseekVocab(t *testing.T) ([]nlp.UnigramPiece, string) {
	t.Helper()
	const bos = "<｜begin▁of▁sentence｜>"
	vocab := []nlp.UnigramPiece{{"<unk>", -20}, {"▁", -3}, {bos, -1}}
	seen := map[string]bool{"<unk>": true, "▁": true, bos: true}
	for _, r := range bos + "ab" { // single-char coverage so Encode always has a path
		if !seen[string(r)] {
			seen[string(r)] = true
			vocab = append(vocab, nlp.UnigramPiece{Piece: string(r), Score: -8})
		}
	}
	return vocab, bos
}

// §B74 tier-1: a special marker whose NAME contains U+2581 must decode VERBATIM.
// DeepSeek's control tokens are literally "<｜begin▁of▁sentence｜>", so a blanket
// ▁→space replacement over the joined pieces hands back "<｜begin of sentence｜>" — a
// corrupted marker — for a completely correct token id. llama.cpp avoids this by
// unescaping only NORMAL tokens and copying CONTROL/USER_DEFINED ones through.
func TestUnigramSpecialMarkerWithSpaceMetaDecodesVerbatim(t *testing.T) {
	vocab, bos := deepseekVocab(t)
	u, err := nlp.NewUnigram(vocab)
	if err != nil {
		t.Fatal(err)
	}
	u.AddSpecialTokens(map[string]int{bos: 2})

	ids := u.EncodeSpecial(bos)
	if len(ids) != 1 || ids[0] != 2 {
		t.Fatalf("EncodeSpecial(%q) = %v, want the single marker id [2]", bos, ids)
	}
	if got := u.Decode(ids); got != bos {
		t.Errorf("Decode(%v) = %q, want the marker verbatim %q — the U+2581 in the "+
			"marker's NAME must not be unescaped to a space", ids, got, bos)
	}
	// §B60 must not be undone: plain Encode still refuses to emit the marker id.
	for _, id := range u.Encode(bos) {
		if id == 2 {
			t.Error("plain Encode emitted the registered control id from literal text")
		}
	}
}

// §B74 tier-1: a literal U+2581 in ORDINARY input cannot be round-tripped — it is
// SentencePiece's escape character and the escaping has no inverse, in the reference
// implementation as much as here. That loss is not fixable without disagreeing with
// SentencePiece/llama.cpp about what "a▁b" tokenizes to, so the contract is that it
// must be DETECTABLE rather than silent.
func TestUnigramSpaceMetaInputIsLossyButDetectable(t *testing.T) {
	u, err := nlp.NewUnigram([]nlp.UnigramPiece{
		{"<unk>", -20}, {"▁", -3}, {"a", -2}, {"b", -2},
	})
	if err != nil {
		t.Fatal(err)
	}
	const withSpace, withMeta = "a b", "a▁b"

	// The predicate is the contract: it flags exactly the input that will not survive.
	if nlp.ContainsSpaceMeta(withSpace) {
		t.Errorf("ContainsSpaceMeta(%q) = true, want false", withSpace)
	}
	if !nlp.ContainsSpaceMeta(withMeta) {
		t.Fatalf("ContainsSpaceMeta(%q) = false — the one lossy input went unreported", withMeta)
	}
	// And it is honest about WHY: the two really are indistinguishable downstream.
	if a, b := u.Encode(withSpace), u.Encode(withMeta); fmt.Sprint(a) != fmt.Sprint(b) {
		t.Fatalf("Encode(%q)=%v and Encode(%q)=%v now differ — if GoAI gained an escape, "+
			"ContainsSpaceMeta and the losslessness godoc must be revised", withSpace, a, withMeta, b)
	}
	if got := u.Decode(u.Encode(withMeta)); got != withSpace {
		t.Errorf("round-trip %q → %q, want %q (the documented lossy result)", withMeta, got, withSpace)
	}
	// Input the predicate clears must actually round-trip.
	if got := u.Decode(u.Encode(withSpace)); got != withSpace {
		t.Errorf("round-trip %q → %q, want %q", withSpace, got, withSpace)
	}
}

// ContainsSpaceMeta flags text that a SentencePiece tokenizer cannot round-trip: ▁
// (U+2581) is the stand-in for a space, so a literal one you pass in is
// indistinguishable from a space and comes back as a space.
func ExampleContainsSpaceMeta() {
	fmt.Println(nlp.ContainsSpaceMeta("hello world"))
	fmt.Println(nlp.ContainsSpaceMeta("hello▁world"))
	fmt.Println(nlp.SpaceMeta)
	// Output:
	// false
	// true
	// ▁
}
