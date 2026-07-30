package nlp_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// TestMLMProtectedArmsAgree pins MLMMaskExcluding's two membership arms against each other.
//
// The arm is chosen purely on len(specialIDs): a short list is scanned, a long one is hashed into a
// set. Both must denote the same protected SET, so the split has to be invisible in the output —
// and it has to be invisible in the RNG SEQUENCE too, which is the sharper requirement here. A
// protected token consumes no randomness, so an arm that disagreed about even one position would
// shift every subsequent draw and change the whole rest of the sequence, not just that token.
//
// The comparison is made exact by padding with DUPLICATES: repeating the id list changes its
// length, and therefore the arm, while leaving the set it denotes untouched. Both runs use freshly
// seeded, identical RNGs, so any difference is attributable to the arm alone.
//
// Verified by mutation rather than asserted: inverting the scan's comparison, making the scan arm
// return false unconditionally, and inverting the MAP arm each redden this, so both paths are
// gated and not merely the one a short list selects.
//
// It was also, briefly, none of those things. This file was first named mlm_arm_test.go, where the
// _arm suffix is a GOARCH build constraint — the file was silently excluded from the package and
// the test never ran at all, while the mutations above were caught by OTHER tests and looked from
// the outside like a working gate.
func TestMLMProtectedArmsAgree(t *testing.T) {
	const seqLen, vocab = 512, 32000
	tokens := make([]int, seqLen)
	for i := range tokens {
		tokens[i] = (i * 7919) % vocab
	}
	// Protect ids the sequence actually contains, so membership fires rather than being vacuously
	// false on both arms.
	base := []int{tokens[3], tokens[64], tokens[130], tokens[400]}
	padded := make([]int, 0, len(base)*8)
	for len(padded) <= 16 { // comfortably past any plausible scan/hash threshold
		padded = append(padded, base...)
	}

	run := func(ids []int) ([]int, []int) {
		rng := rand.New(rand.NewPCG(31, 17))
		return nlp.MLMMaskExcluding(tokens, 0.5, 103, vocab, ids, rng)
	}
	scanIn, scanLab := run(base)
	mapIn, mapLab := run(padded)

	for i := range scanIn {
		if scanIn[i] != mapIn[i] {
			t.Fatalf("input[%d]: scan arm %d, map arm %d — the membership paths disagree, which also "+
				"means the RNG sequences diverged", i, scanIn[i], mapIn[i])
		}
		if scanLab[i] != mapLab[i] {
			t.Fatalf("labels[%d]: scan arm %d, map arm %d", i, scanLab[i], mapLab[i])
		}
	}

	// Anti-vacuity: the protected positions must actually be protected, or both arms agreed on a
	// predicate that is false everywhere and the comparison proved nothing.
	protected := map[int]bool{}
	for _, id := range base {
		protected[id] = true
	}
	var guarded, predicted int
	for i, tok := range tokens {
		if protected[tok] {
			guarded++
			if scanLab[i] != nlp.MLMIgnoreLabel {
				t.Fatalf("token %d at position %d is protected but was selected for prediction", tok, i)
			}
		} else if scanLab[i] != nlp.MLMIgnoreLabel {
			predicted++
		}
	}
	if guarded == 0 {
		t.Fatal("no position held a protected id; the fixture never exercises membership")
	}
	if predicted == 0 {
		t.Fatal("no position was selected for prediction; the fixture never exercises the masking path")
	}
}
