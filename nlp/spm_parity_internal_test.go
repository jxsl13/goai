package nlp

import (
	"math/rand"
	"strings"
	"testing"
)

// spmBoundsNaive is the pre-optimization O(N²) full-rescan merge — the reference the O(N log N)
// heap-based spmBounds must match token-for-token (§SPM merge parity). Kept ONLY as a test oracle.
func spmBoundsNaive(s *SPM, str string) []int {
	bounds := make([]int, 0, len(str)+1)
	for i := range str {
		bounds = append(bounds, i)
	}
	bounds = append(bounds, len(str))
	for {
		best, bestScore := -1, 0.0
		for i := 0; i+2 < len(bounds); i++ {
			merged := str[bounds[i]:bounds[i+2]]
			id, ok := s.id[merged]
			if ok && s.specials.blocked(merged) {
				continue
			}
			if ok && (best < 0 || s.pieces[id].Score > bestScore) {
				best, bestScore = i, s.pieces[id].Score
			}
		}
		if best < 0 {
			break
		}
		bounds = append(bounds[:best+1], bounds[best+2:]...)
	}
	return bounds
}

// TestSPMBoundsParityVsNaive guards the heap merge (spmBounds) against the naive reference across
// random inputs — INCLUDING many equal scores, which stress the leftmost-wins tie-break the heap
// key (score desc, left-offset asc) must reproduce exactly. A mismatch means the fast path diverged.
func TestSPMBoundsParityVsNaive(t *testing.T) {
	var vocab []UnigramPiece
	vocab = append(vocab, UnigramPiece{Piece: "<unk>", Score: 0}, UnigramPiece{Piece: "▁", Score: 0})
	for c := 'a'; c <= 'z'; c++ {
		vocab = append(vocab, UnigramPiece{Piece: string(c), Score: 0})
	}
	// pieces with MANY tied scores (%5) to exercise tie-breaking, plus multi-char and ▁-prefixed.
	ps := []string{"th", "the", "▁the", "in", "ing", "er", "▁a", "at", "he", "▁t", "an", "re", "on", "▁th", "ha", "▁he", "ca", "sa", "ma", "▁cat", "▁sat"}
	for i, p := range ps {
		vocab = append(vocab, UnigramPiece{Piece: p, Score: -float64(i % 5)})
	}
	s, err := NewSPM(vocab)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	letters := "abcdefghijklmnopqrstuvwxyz    " // extra spaces → ▁ runs
	for it := 0; it < 5000; it++ {
		var b strings.Builder
		for j := 0; j < rng.Intn(48); j++ {
			b.WriteByte(letters[rng.Intn(len(letters))])
		}
		str := spaceMeta + strings.ReplaceAll(b.String(), " ", spaceMeta)
		got, want := s.spmBounds(str), spmBoundsNaive(s, str)
		if len(got) != len(want) {
			t.Fatalf("bounds len %d != %d for %q\n got=%v\nwant=%v", len(got), len(want), str, got, want)
		}
		for k := range got {
			if got[k] != want[k] {
				t.Fatalf("bounds[%d] %d != %d for %q\n got=%v\nwant=%v", k, got[k], want[k], str, got, want)
			}
		}
	}
}
