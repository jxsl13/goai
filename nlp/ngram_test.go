package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

func zeros(n int) []float64 { return make([]float64, n) }

// §V16 tier-1 (rule CONFIRMED, HF NoRepeatNGramLogitsProcessor): the token that would
// complete a previously-seen n-gram (given the current last-(n−1) context) is set to
// −∞; all others are untouched.
func TestNoRepeatNGramTrigram(t *testing.T) {
	// seq "1 2 3 1 2": context = last 2 = [1 2]; completing "1 2 3" again with 3 is banned.
	logits := zeros(5)
	nlp.NoRepeatNGramBlock(logits, []int{1, 2, 3, 1, 2}, 3)
	if !math.IsInf(logits[3], -1) {
		t.Errorf("token 3 not blocked: %v", logits)
	}
	for _, tok := range []int{0, 1, 2, 4} {
		if logits[tok] != 0 {
			t.Errorf("token %d wrongly changed to %g", tok, logits[tok])
		}
	}
}

// §V16 tier-1: bigram (n=2) bans every token that has ever directly followed the last
// token.
func TestNoRepeatNGramBigramMultiple(t *testing.T) {
	// seq "0 1 0 2 0": last token 0 has been followed by 1 and by 2 ⇒ both banned.
	logits := zeros(4)
	nlp.NoRepeatNGramBlock(logits, []int{0, 1, 0, 2, 0}, 2)
	if !math.IsInf(logits[1], -1) || !math.IsInf(logits[2], -1) {
		t.Errorf("tokens 1,2 not both blocked: %v", logits)
	}
	if logits[0] != 0 || logits[3] != 0 {
		t.Errorf("tokens 0,3 wrongly changed: %v", logits)
	}
}

// §V16 tier-1: no-op when the sequence is shorter than n, or n ≤ 0.
func TestNoRepeatNGramNoop(t *testing.T) {
	for _, tc := range []struct {
		seq []int
		n   int
	}{
		{[]int{1, 2}, 3},       // L < n
		{[]int{1, 2, 3, 1}, 0}, // disabled
		{nil, 3},               // empty
	} {
		logits := zeros(4)
		nlp.NoRepeatNGramBlock(logits, tc.seq, tc.n)
		for i, v := range logits {
			if v != 0 {
				t.Errorf("seq=%v n=%d: token %d changed to %g, want no-op", tc.seq, tc.n, i, v)
			}
		}
	}
}

// §V2: a completing token outside the logits range is ignored (no panic, no change).
func TestNoRepeatNGramOutOfRange(t *testing.T) {
	logits := zeros(3)                                // vocab 0..2
	nlp.NoRepeatNGramBlock(logits, []int{0, 5, 0}, 2) // token after [0] is 5 (≥ vocab)
	for i, v := range logits {
		if v != 0 {
			t.Errorf("token %d changed to %g, want unchanged (banned token out of range)", i, v)
		}
	}
}

func ExampleNoRepeatNGramBlock() {
	// "the cat sat the cat": with n=3 the context "the cat" was already followed by
	// "sat", so repeating "sat" (token 2) is blocked.
	logits := []float64{0, 0, 0, 0} // the=0 cat=1 sat=2 mat=3
	nlp.NoRepeatNGramBlock(logits, []int{0, 1, 2, 0, 1}, 3)
	fmt.Printf("sat blocked=%v mat free=%v\n", math.IsInf(logits[2], -1), logits[3] == 0)
	// Output:
	// sat blocked=true mat free=true
}
