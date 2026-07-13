package nlp_test

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

func mustWP(t *testing.T, vocab []string, opts ...nlp.WordPieceOption) *nlp.WordPiece {
	t.Helper()
	w, err := nlp.NewWordPiece(vocab, opts...)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// §V16 tier-1 (greedy longest-match-first + ## continuation): the canonical example — "playing"
// with {play, ##ing} tokenizes to [play, ##ing].
func TestWordPiecePlaying(t *testing.T) {
	w := mustWP(t, []string{"[UNK]", "play", "##ing"})
	if got := w.Encode("playing"); !slices.Equal(got, []int{1, 2}) {
		t.Errorf("Encode(playing) = %v, want [1 2] (play,##ing)", got)
	}
}

// §V16 tier-1 (MaxMatch): "unaffable" with {un, ##aff, ##able} → [un, ##aff, ##able].
func TestWordPieceMaxMatch(t *testing.T) {
	w := mustWP(t, []string{"[UNK]", "un", "##aff", "##able"})
	if got := w.Encode("unaffable"); !slices.Equal(got, []int{1, 2, 3}) {
		t.Errorf("Encode(unaffable) = %v, want [1 2 3]", got)
	}
}

// §V16 tier-1 (LONGEST match preferred): with both "un" and "una" available the encoder takes the
// longer "una" first — proving longest-match, not shortest.
func TestWordPieceLongestPreferred(t *testing.T) {
	w := mustWP(t, []string{"[UNK]", "un", "una", "##ffable"})
	got := w.Encode("unaffable")
	if !slices.Equal(got, []int{2, 3}) { // una, ##ffable
		t.Errorf("Encode(unaffable) = %v, want [2 3] (una,##ffable — longest first)", got)
	}
}

// §V16 tier-1 (whole-word UNK): if any position cannot be matched down to a single char, the WHOLE
// word becomes one [UNK] — not a partial tokenization of the matched prefix.
func TestWordPieceWholeWordUnk(t *testing.T) {
	w := mustWP(t, []string{"[UNK]", "un", "##aff"}) // no "##able" ⇒ fails after ##aff
	if got := w.Encode("unaffable"); !slices.Equal(got, []int{0}) {
		t.Errorf("Encode(unaffable) = %v, want [0] (whole-word UNK)", got)
	}
}

// §V16 tier-1 (max-chars): a word longer than MaxChars is emitted as UNK without tokenizing.
func TestWordPieceMaxChars(t *testing.T) {
	w := mustWP(t, []string{"[UNK]", "a", "##b", "##c", "##d"}, nlp.WithWordPieceMaxChars(3))
	if got := w.Encode("abcd"); !slices.Equal(got, []int{0}) {
		t.Errorf("Encode(abcd) with maxChars=3 = %v, want [0] (UNK)", got)
	}
	if got := w.Encode("abc"); !slices.Equal(got, []int{1, 2, 3}) { // within the limit
		t.Errorf("Encode(abc) = %v, want [1 2 3]", got)
	}
}

// §V2 (multi-word): each whitespace-separated word is tokenized independently.
func TestWordPieceMultiWord(t *testing.T) {
	w := mustWP(t, []string{"[UNK]", "play", "##ing", "game", "##s"})
	if got := w.Encode("playing games"); !slices.Equal(got, []int{1, 2, 3, 4}) {
		t.Errorf("Encode(playing games) = %v, want [1 2 3 4]", got)
	}
}

// §V15 (round-trip): with full single-character coverage, Decode∘Encode recovers the text up to
// whitespace normalization (word boundaries preserved via ## vs space).
func TestWordPieceRoundTrip(t *testing.T) {
	w := mustWP(t, charVocab())
	for _, text := range []string{"ab cde a", "a", "edcba abc", "  a   bc  "} {
		norm := strings.Join(strings.Fields(text), " ")
		if got := w.Decode(w.Encode(text)); got != norm {
			t.Errorf("round-trip %q ⇒ %q, want %q", text, got, norm)
		}
	}
}

func TestWordPieceErrors(t *testing.T) {
	if _, err := nlp.NewWordPiece(nil); err == nil {
		t.Error("empty vocab: want error")
	}
	if _, err := nlp.NewWordPiece([]string{"a", "a"}); err == nil {
		t.Error("duplicate piece: want error")
	}
}

// charVocab is [UNK] + every letter a–e in bare and "##" continuation form.
func charVocab() []string {
	v := []string{"[UNK]"}
	for c := 'a'; c <= 'e'; c++ {
		v = append(v, string(c))
	}
	for c := 'a'; c <= 'e'; c++ {
		v = append(v, "##"+string(c))
	}
	return v
}

// §V15 fuzz: over a controlled alphabet with full single-char coverage, WordPiece round-trips to
// the whitespace-normalized text.
func FuzzWordPieceRoundTrip(f *testing.F) {
	w, _ := nlp.NewWordPiece(charVocab())
	f.Add([]byte("abc de"))
	f.Add([]byte(""))
	f.Add([]byte{0, 1, 2, 5, 5, 3})
	f.Fuzz(func(t *testing.T, data []byte) {
		const alpha = "abcde " // last symbol is the space separator
		var sb strings.Builder
		for _, b := range data {
			sb.WriteByte(alpha[int(b)%len(alpha)])
		}
		text := sb.String()
		// honest expectation: fully-covered words round-trip; a word longer than maxChars (100)
		// is lossily emitted as [UNK] by design.
		words := strings.Fields(text)
		for i, wd := range words {
			if len([]rune(wd)) > 100 {
				words[i] = "[UNK]"
			}
		}
		want := strings.Join(words, " ")
		if got := w.Decode(w.Encode(text)); got != want {
			t.Fatalf("round-trip %q ⇒ %q, want %q", text, got, want)
		}
	})
}

func ExampleWordPiece() {
	w, _ := nlp.NewWordPiece([]string{"[UNK]", "play", "##ing", "un", "##able"})
	fmt.Println(w.Encode("playing"))
	fmt.Println(w.Decode(w.Encode("playing")))
	// Output:
	// [1 2]
	// playing
}
