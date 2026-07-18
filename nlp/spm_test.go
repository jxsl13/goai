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

// §B74 tier-1: the SPM twin of the Unigram case — a control token whose NAME contains
// U+2581 decodes verbatim, while a literal U+2581 in ordinary text stays lossy and
// detectable. SPM's Decode also expands <0xNN> byte pieces, and those bytes mean
// themselves: llama.cpp unescapes whitespace for NORMAL tokens only, never for BYTE or
// CONTROL/USER_DEFINED ones.
func TestSPMSpaceMetaHandling(t *testing.T) {
	const bos = "<｜begin▁of▁sentence｜>"
	vocab := []nlp.UnigramPiece{{"<unk>", -20}, {"▁", -3}, {bos, -1}}
	seen := map[string]bool{"<unk>": true, "▁": true, bos: true}
	for _, r := range bos + "ab" {
		if !seen[string(r)] {
			seen[string(r)] = true
			vocab = append(vocab, nlp.UnigramPiece{Piece: string(r), Score: -8})
		}
	}
	s, err := nlp.NewSPM(vocab)
	if err != nil {
		t.Fatal(err)
	}
	s.AddSpecialTokens(map[string]int{bos: 2})

	t.Run("marker decodes verbatim", func(t *testing.T) {
		ids := s.EncodeSpecial(bos)
		if len(ids) != 1 || ids[0] != 2 {
			t.Fatalf("EncodeSpecial(%q) = %v, want [2]", bos, ids)
		}
		if got := s.Decode(ids); got != bos {
			t.Errorf("Decode(%v) = %q, want %q verbatim — U+2581 in a control token's "+
				"NAME must not be unescaped", ids, got, bos)
		}
	})

	t.Run("literal U+2581 is lossy but detectable", func(t *testing.T) {
		if !nlp.ContainsSpaceMeta("a▁b") {
			t.Fatal(`ContainsSpaceMeta("a▁b") = false — the lossy input went unreported`)
		}
		if nlp.ContainsSpaceMeta("a b") {
			t.Error(`ContainsSpaceMeta("a b") = true, want false`)
		}
		if got := s.Decode(s.Encode("a▁b")); got != "a b" {
			t.Errorf(`round-trip "a▁b" → %q, want "a b" (the documented lossy result)`, got)
		}
	})

	t.Run("ordinary text still round-trips", func(t *testing.T) {
		for _, in := range []string{"ab", "a b", "b a"} {
			if got := s.Decode(s.Encode(in)); got != in {
				t.Errorf("round-trip %q → %q", in, got)
			}
		}
	})
}
