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
