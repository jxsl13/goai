package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// llama-family GGUF scores are negative merge ranks, not log-probabilities. Under
// Viterbi (Unigram) the sum −61+−6 for ▁T+he beats ▁The's −156 and the encoding
// fragments (§B59); the SPM merge encoder must produce the whole-word piece, matching
// llama.cpp. The scores below are the REAL Mistral-7B vocabulary entries.
func TestSPMRankScoresPickWholeWord(t *testing.T) {
	vocab := []nlp.UnigramPiece{
		{Piece: "<unk>", Score: 0},
		{Piece: "▁", Score: -100},
		{Piece: "T", Score: -101},
		{Piece: "h", Score: -102},
		{Piece: "e", Score: -103},
		{Piece: "he", Score: -6},
		{Piece: "▁T", Score: -61},
		{Piece: "▁The", Score: -156},
	}
	s, err := nlp.NewSPM(vocab)
	if err != nil {
		t.Fatal(err)
	}
	ids := s.Encode("The")
	if len(ids) != 1 || vocab[ids[0]].Piece != "▁The" {
		got := make([]string, len(ids))
		for i, id := range ids {
			got[i] = vocab[id].Piece
		}
		t.Fatalf("SPM must merge to the whole-word piece ▁The, got %v", got)
	}
	// The same vocabulary under Unigram Viterbi fragments — the documented divergence
	// this encoder exists to avoid.
	u, err := nlp.NewUnigram(vocab)
	if err != nil {
		t.Fatal(err)
	}
	if v := u.Encode("The"); len(v) == 1 {
		t.Fatalf("premise broken: Viterbi unexpectedly also picked the whole word (%v)", v)
	}
}

// Merge order follows scores (higher merges first), ties go to the leftmost pair, and
// decoding restores the original text including the ▁→space mapping.
func TestSPMEncodeDecodeRoundTrip(t *testing.T) {
	vocab := []nlp.UnigramPiece{
		{Piece: "<unk>", Score: 0},
		{Piece: "▁", Score: -3},
		{Piece: "a", Score: -1},
		{Piece: "b", Score: -2},
		{Piece: "ab", Score: -4},
		{Piece: "▁ab", Score: -5},
	}
	s, err := nlp.NewSPM(vocab)
	if err != nil {
		t.Fatal(err)
	}
	ids := s.Encode("ab ab")
	if got := s.Decode(ids); got != "ab ab" {
		t.Fatalf("round trip: got %q, want %q", got, "ab ab")
	}
}

// A character no piece covers must fall back to the <0xNN> byte pieces (llama.cpp
// semantics), and Decode must restore the raw bytes.
func TestSPMByteFallback(t *testing.T) {
	vocab := []nlp.UnigramPiece{
		{Piece: "<unk>", Score: 0},
		{Piece: "▁", Score: -1},
		{Piece: "<0xE2>", Score: -1000},
		{Piece: "<0x82>", Score: -1000},
		{Piece: "<0xAC>", Score: -1000},
	}
	s, err := nlp.NewSPM(vocab)
	if err != nil {
		t.Fatal(err)
	}
	ids := s.Encode("€") // U+20AC = E2 82 AC, not in vocab as a piece
	if len(ids) != 4 {   // dummy-prefix ▁ + 3 byte tokens
		t.Fatalf("expected ▁ + 3 byte-fallback tokens for €, got %v", ids)
	}
	if got := s.Decode(ids); got != "€" {
		t.Fatalf("byte round trip: got %q, want %q", got, "€")
	}
}
