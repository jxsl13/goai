package nlp_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

const wpTokJSON = `{
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {"[UNK]": 0, "play": 1, "##ing": 2, "game": 3, "##s": 4}
  }
}`

// §V15 / §V16-exempt (definitional format): a WordPiece tokenizer.json parses into a working
// tokenizer with the file's continuation prefix and unk token.
func TestWordPieceFromJSONParse(t *testing.T) {
	w, err := nlp.WordPieceFromJSON([]byte(wpTokJSON))
	if err != nil {
		t.Fatal(err)
	}
	if got := w.Encode("playing games"); !slices.Equal(got, []int{1, 2, 3, 4}) {
		t.Errorf("Encode(playing games) = %v, want [1 2 3 4]", got)
	}
	if got := w.Encode("zzz"); !slices.Equal(got, []int{0}) { // uncoverable ⇒ [UNK]=0
		t.Errorf("Encode(zzz) = %v, want [0] ([UNK])", got)
	}
}

// §V15 (round-trip): ToJSON→FromJSON preserves the tokenizer (identical Encode + config).
func TestWordPieceJSONRoundTrip(t *testing.T) {
	orig, err := nlp.NewWordPiece([]string{"[UNK]", "un", "##aff", "##able", "play", "##ing"},
		nlp.WithWordPieceMaxChars(64))
	if err != nil {
		t.Fatal(err)
	}
	data, err := orig.ToJSON()
	if err != nil {
		t.Fatal(err)
	}
	back, err := nlp.WordPieceFromJSON(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"unaffable", "playing", "un ##aff", "xyz", "playing unaffable"} {
		if a, b := orig.Encode(s), back.Encode(s); !slices.Equal(a, b) {
			t.Errorf("round-trip Encode(%q): %v != %v", s, a, b)
		}
	}
}

// §V15: added_tokens share the id space with the vocab.
func TestWordPieceJSONAddedTokens(t *testing.T) {
	js := `{"model":{"type":"WordPiece","vocab":{"[UNK]":0,"a":1,"##b":2}},
	        "added_tokens":[{"id":3,"content":"[SEP]"}]}`
	w, err := nlp.WordPieceFromJSON([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	if got := w.Decode([]int{3}); got != "[SEP]" {
		t.Errorf("added token id 3 = %q, want [SEP]", got)
	}
}

func TestWordPieceFromJSONErrors(t *testing.T) {
	if _, err := nlp.WordPieceFromJSON([]byte("{not json")); err == nil {
		t.Error("invalid JSON: want error")
	}
	if _, err := nlp.WordPieceFromJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0}}}`)); err == nil {
		t.Error("non-WordPiece type: want error")
	}
	if _, err := nlp.WordPieceFromJSON([]byte(`{"model":{"type":"WordPiece","vocab":{}}}`)); err == nil {
		t.Error("empty vocab: want error")
	}
}

// §V15 fuzz: WordPieceFromJSON never panics on arbitrary bytes; any tokenizer it produces
// re-round-trips through ToJSON.
func FuzzWordPieceFromJSON(f *testing.F) {
	f.Add([]byte(wpTokJSON))
	f.Add([]byte(`{"model":{"type":"WordPiece","vocab":{"[UNK]":0,"a":1}}}`))
	f.Add([]byte("{}"))
	f.Add([]byte(""))
	f.Fuzz(func(t *testing.T, data []byte) {
		w, err := nlp.WordPieceFromJSON(data)
		if err != nil {
			return // parse rejected — fine, as long as it did not panic
		}
		out, err := w.ToJSON()
		if err != nil {
			t.Fatalf("ToJSON failed for a parsed tokenizer: %v", err)
		}
		w2, err := nlp.WordPieceFromJSON(out)
		if err != nil {
			t.Fatalf("re-parse of ToJSON failed: %v", err)
		}
		for _, s := range []string{"hello", "playing", "a b c"} {
			if a, b := w.Encode(s), w2.Encode(s); !slices.Equal(a, b) {
				t.Fatalf("round-trip Encode(%q): %v != %v", s, a, b)
			}
		}
	})
}

func ExampleWordPieceFromJSON() {
	js := `{"model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##",
	        "vocab":{"[UNK]":0,"play":1,"##ing":2}}}`
	w, _ := nlp.WordPieceFromJSON([]byte(js))
	fmt.Println(w.Encode("playing"))
	// Output:
	// [1 2]
}

// §B47 (same class as the BPE parser): implausible token ids must be rejected,
// not allocated.
func TestWordPieceFromJSONRejectsImplausibleID(t *testing.T) {
	hostile := []byte(`{"model":{"type":"WordPiece","unk_token":"[UNK]","vocab":{"[UNK]":0,"y":9999990000009}}}`)
	if _, err := nlp.WordPieceFromJSON(hostile); err == nil {
		t.Fatal("implausible token id must error")
	}
}
