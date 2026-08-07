package nlp

import (
	"reflect"
	"testing"
)

// TestSuffixLookupPicksMajority verifies the drafter behaves as SuffixDecoding intends: when the longest
// recurring suffix has SEVERAL earlier occurrences with different continuations, it drafts the majority
// (dominant) continuation, whereas NgramLookup copies the arbitrary earliest occurrence. On a history
// where the dominant continuation is what the model will actually produce, this is a strictly longer
// accepted draft.
func TestSuffixLookupPicksMajority(t *testing.T) {
	// The FULL maxNgram=3 suffix "9 9 9" itself recurs with different continuations: earliest → NOISE
	// [1 1], then three times → DOMINANT [7 7]. Current tail is "9 9 9" → majority wants [7 7], not [1 1].
	seq := []int{
		9, 9, 9, 1, 1, // earliest occ of "9 9 9" → noise cont [1 1]
		9, 9, 9, 7, 7, // dominant occ #1 → [7 7]
		9, 9, 9, 7, 7, // dominant occ #2
		9, 9, 9, 7, 7, // dominant occ #3
		9, 9, 9, // current suffix
	}
	ng := NgramLookup(seq, 3, 3)
	sd := SuffixLookup(seq, 3, 3)
	t.Logf("NgramLookup draft=%v  SuffixLookup draft=%v", ng, sd)
	if len(ng) == 0 || ng[0] != 1 {
		t.Errorf("NgramLookup expected to copy earliest occ [1 ...], got %v", ng)
	}
	if len(sd) < 2 || sd[0] != 7 || sd[1] != 7 {
		t.Errorf("SuffixLookup expected majority continuation [7 7 ...], got %v", sd)
	}
}

// TestSuffixLookupMatchesNgramWhenUnique confirms that with a single occurrence (no ambiguity) the two
// drafters agree — SuffixLookup is a strict superset, never worse on the simple-repetition case.
func TestSuffixLookupMatchesNgramWhenUnique(t *testing.T) {
	seq := []int{3, 4, 5, 6, 7, 3, 4} // suffix "3 4" occurs once earlier (start 0) → cont [5 6 7 3]
	ng := NgramLookup(seq, 3, 4)
	sd := SuffixLookup(seq, 3, 4)
	if !reflect.DeepEqual(ng, sd) {
		t.Errorf("unique-occurrence drafts should match: ngram=%v suffix=%v", ng, sd)
	}
	if !reflect.DeepEqual(sd, []int{5, 6, 7, 3}) {
		t.Errorf("expected [5 6 7 3], got %v", sd)
	}
}

// TestSuffixLookupEmpty: no recurring suffix → nil (falls back to a plain step, like NgramLookup).
func TestSuffixLookupEmpty(t *testing.T) {
	if d := SuffixLookup([]int{1, 2, 3, 4, 5}, 3, 4); d != nil {
		t.Errorf("expected nil for non-recurring seq, got %v", d)
	}
}
