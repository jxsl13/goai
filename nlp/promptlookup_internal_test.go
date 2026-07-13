package nlp

import (
	"fmt"
	"testing"
)

// §R89: the prompt-lookup drafter copies the continuation of an earlier occurrence of
// the current suffix. It prefers the longest n-gram, returns at most draftLen tokens,
// and yields nil when nothing recurs.
func TestNgramLookup(t *testing.T) {
	cases := []struct {
		name               string
		seq                []int
		maxNgram, draftLen int
		want               []int
	}{
		// suffix [2,3] recurs (seq[1:3]); the tokens after that occurrence are [4,5,...]
		{"bigram", []int{2, 3, 4, 5, 6, 2, 3}, 2, 3, []int{4, 5, 6}},
		{"draftLen caps", []int{2, 3, 4, 5, 6, 2, 3}, 2, 1, []int{4}},
		// longest match preferred: suffix [8,2,3] recurs? no; falls to [2,3]
		{"prefer-longer", []int{1, 2, 3, 9, 1, 2, 3}, 3, 2, []int{9, 1}}, // [1,2,3] earlier → follows 9,1
		{"no recurrence", []int{1, 2, 3, 4, 5}, 3, 4, nil},
		{"too short", []int{7}, 3, 4, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := NgramLookup(c.seq, c.maxNgram, c.draftLen)
			if fmt.Sprint(got) != fmt.Sprint(c.want) {
				t.Errorf("NgramLookup(%v, %d, %d) = %v, want %v", c.seq, c.maxNgram, c.draftLen, got, c.want)
			}
		})
	}
}
