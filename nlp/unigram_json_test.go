package nlp_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// a tiny Unigram tokenizer.json: <unk>, the ▁ space meta, single chars, and a merged
// "▁hi" piece scored higher than its 3-piece split.
const sampleUnigramJSON = `{
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["▁", -3.0],
      ["h", -4.0],
      ["i", -4.0],
      ["▁hi", -5.0]
    ]
  }
}`

// §V16 tier-1 (DEFINITIONAL, HF tokenizer.json schema): UnigramFromJSON reads the scored
// vocab and unk_id, and Viterbi picks the higher-scoring "▁hi" over "▁"+"h"+"i".
func TestUnigramFromJSONParse(t *testing.T) {
	u, err := nlp.UnigramFromJSON([]byte(sampleUnigramJSON))
	if err != nil {
		t.Fatalf("UnigramFromJSON: %v", err)
	}
	if ids := u.Encode("hi"); len(ids) != 1 || ids[0] != 4 {
		t.Errorf("Encode(hi) = %v, want [4]", ids)
	}
	if got := u.Decode([]int{4}); got != "hi" {
		t.Errorf("Decode(4) = %q, want hi", got)
	}
}

// §B108: an added token whose id sits PAST the base vocab (how HuggingFace chat models
// append control tokens like <|im_start|>) must survive Encode→Decode. EncodeSpecial
// correctly emitted id 5, but Decode indexed only pieces[0..4] and silently DROPPED it —
// a round-trip loss on every such model. Now decoded via the specials' id→text reverse.
func TestUnigramAddedTokenBeyondVocabDecode(t *testing.T) {
	const j = `{
	  "model": { "type": "Unigram", "unk_id": 0, "vocab": [
	    ["<unk>", 0.0], ["▁", -3.0], ["h", -4.0], ["i", -4.0], ["▁hi", -5.0]
	  ]},
	  "added_tokens": [ {"id": 5, "content": "<|im_start|>", "special": true} ]
	}`
	u, err := nlp.UnigramFromJSON([]byte(j))
	if err != nil {
		t.Fatal(err)
	}
	// Pure decode: id 5 is past pieces[0..4] and used to be dropped → "".
	if got := u.Decode([]int{5}); got != "<|im_start|>" {
		t.Errorf("Decode([5]) = %q, want %q (added token past the vocab was dropped)", got, "<|im_start|>")
	}
	// Clean round-trip with the special LAST (the leading dummy-prefix space is trimmed):
	// EncodeSpecial emits id 5, and Decode must round-trip it rather than silently drop it.
	ids := u.EncodeSpecial("hi<|im_start|>")
	if got := u.Decode(ids); got != "hi<|im_start|>" {
		t.Errorf("Decode(%v) = %q, want %q", ids, got, "hi<|im_start|>")
	}
}

// §V15: round-trip — load → ToJSON → load reproduces identical encodings.
func TestUnigramJSONRoundTrip(t *testing.T) {
	u, err := nlp.UnigramFromJSON([]byte(sampleUnigramJSON))
	if err != nil {
		t.Fatal(err)
	}
	data, err := u.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	u2, err := nlp.UnigramFromJSON(data)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	for _, s := range []string{"hi", "h", "i", "ihi", "hih"} {
		a, b := u.Encode(s), u2.Encode(s)
		if len(a) != len(b) {
			t.Errorf("Encode(%q): %v vs %v (len)", s, a, b)
			continue
		}
		for i := range a {
			if a[i] != b[i] {
				t.Errorf("Encode(%q): %v vs %v", s, a, b)
				break
			}
		}
	}
	// a second serialize is byte-stable in length.
	data2, _ := u2.ToJSON()
	if len(data) != len(data2) {
		t.Errorf("ToJSON not stable: %d vs %d bytes", len(data), len(data2))
	}
}

func TestUnigramFromJSONErrors(t *testing.T) {
	if _, err := nlp.UnigramFromJSON([]byte(`{bad`)); err == nil {
		t.Error("malformed JSON: want error")
	}
	if _, err := nlp.UnigramFromJSON([]byte(`{"model":{"type":"BPE","vocab":[]}}`)); err == nil {
		t.Error("non-Unigram type: want error")
	}
	if _, err := nlp.UnigramFromJSON([]byte(`{"model":{"type":"Unigram","vocab":[]}}`)); err == nil {
		t.Error("empty vocab: want error")
	}
	if _, err := nlp.UnigramFromJSON([]byte(`{"model":{"type":"Unigram","vocab":[["a"]]}}`)); err == nil {
		t.Error("malformed vocab entry: want error")
	}
}

// §V15: fuzzing the loader never panics — arbitrary bytes return an error or a tokenizer.
func FuzzUnigramFromJSON(f *testing.F) {
	f.Add([]byte(sampleUnigramJSON))
	f.Add([]byte(`{"model":{"type":"Unigram","unk_id":0,"vocab":[["a",-1.0]]}}`))
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"model":{"type":"Unigram","vocab":[["a","notnum"]]}}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		u, err := nlp.UnigramFromJSON(data)
		if err == nil && u != nil {
			_ = u.Encode("hello world")
			if out, e := u.ToJSON(); e == nil {
				_, _ = nlp.UnigramFromJSON(out)
			}
		}
	})
}

func ExampleUnigramFromJSON() {
	j := `{"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["▁",-3.0],["h",-4.0],["i",-4.0],["▁hi",-5.0]]}}`
	u, _ := nlp.UnigramFromJSON([]byte(j))
	fmt.Println(u.Encode("hi"))
	// Output:
	// [4]
}
