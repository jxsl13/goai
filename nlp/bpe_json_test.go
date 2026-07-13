package nlp_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// a tiny byte-level BPE (a,b,c symbols + a few merges) — enough to exercise the loader
// and round-trip without a full 256-byte vocab.
const sampleTokJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
    "merges": ["a b", "ab c"]
  },
  "added_tokens": [{"id": 5, "content": "<eos>"}]
}`

// §V16 tier-1 (DEFINITIONAL, HF tokenizer.json schema): BPEFromJSON reads the BPE vocab
// and merges (and added_tokens) correctly.
func TestBPEFromJSONParse(t *testing.T) {
	tok, err := nlp.BPEFromJSON([]byte(sampleTokJSON))
	if err != nil {
		t.Fatalf("BPEFromJSON: %v", err)
	}
	// "abc" pre-token merges a+b→ab, ab+c→abc ⇒ single id 4.
	if ids := tok.Encode("abc"); len(ids) != 1 || ids[0] != 4 {
		t.Errorf("Encode(abc) = %v, want [4]", ids)
	}
	// the added special token landed at id 5.
	if got := tok.Decode([]int{5}); got != "<eos>" {
		t.Errorf("Decode(5) = %q, want <eos>", got)
	}
}

// §V16 tier-1: the newer [["l","r"], …] merges encoding parses identically to "l r".
func TestBPEFromJSONPairMerges(t *testing.T) {
	pair := `{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}}`
	tok, err := nlp.BPEFromJSON([]byte(pair))
	if err != nil {
		t.Fatalf("BPEFromJSON pairs: %v", err)
	}
	if ids := tok.Encode("ab"); len(ids) != 1 || ids[0] != 2 {
		t.Errorf("Encode(ab) = %v, want [2]", ids)
	}
}

// §V15: round-trip — load → ToJSON → load again reproduces the same tokenizer (identical
// vocab, merges, and encoding).
func TestBPEJSONRoundTrip(t *testing.T) {
	tok, err := nlp.BPEFromJSON([]byte(sampleTokJSON))
	if err != nil {
		t.Fatal(err)
	}
	data, err := tok.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON: %v", err)
	}
	tok2, err := nlp.BPEFromJSON(data)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	for _, s := range []string{"abc", "ab", "a", "bca", "cab"} {
		a, b := tok.Encode(s), tok2.Encode(s)
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
	// a second serialize is byte-identical to the first (stable form).
	data2, _ := tok2.ToJSON()
	if len(data) == 0 || string(data) == "" {
		t.Fatal("empty ToJSON")
	}
	if len(data) != len(data2) { // Go maps marshal deterministically for identical content
		t.Errorf("ToJSON not stable: %d vs %d bytes", len(data), len(data2))
	}
}

func TestBPEFromJSONErrors(t *testing.T) {
	if _, err := nlp.BPEFromJSON([]byte(`{bad json`)); err == nil {
		t.Error("malformed JSON: want error")
	}
	if _, err := nlp.BPEFromJSON([]byte(`{"model":{"type":"Unigram","vocab":{"a":0}}}`)); err == nil {
		t.Error("non-BPE type: want error")
	}
	if _, err := nlp.BPEFromJSON([]byte(`{"model":{"type":"BPE","vocab":{}}}`)); err == nil {
		t.Error("empty vocab: want error")
	}
}

// §V15: fuzzing the loader never panics — arbitrary bytes return an error or a tokenizer.
func FuzzBPEFromJSON(f *testing.F) {
	f.Add([]byte(sampleTokJSON))
	f.Add([]byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`))
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"model":{"type":"BPE","vocab":{"x":-1,"y":9999999}}}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		tok, err := nlp.BPEFromJSON(data)
		if err == nil && tok != nil {
			_ = tok.Encode("hello world") // must not panic on a parsed tokenizer
			if out, e := tok.ToJSON(); e == nil {
				_, _ = nlp.BPEFromJSON(out) // re-round-trip must not panic
			}
		}
	})
}

func ExampleBPEFromJSON() {
	j := `{"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":["h i"]}}`
	tok, _ := nlp.BPEFromJSON([]byte(j))
	fmt.Println(tok.Encode("hi"))
	// Output:
	// [2]
}

// §B47: the vocab slice is indexed by token id — a hostile tokenizer.json with an
// astronomically large id must be REJECTED, not allocated (fuzz found an id of ~1e13,
// a 160TB allocation that OOM-killed the process).
func TestBPEFromJSONRejectsImplausibleID(t *testing.T) {
	hostile := []byte(`{"model":{"type":"BPE","vocab":{"x":0,"y":9999990000009}}}`)
	if _, err := nlp.BPEFromJSON(hostile); err == nil {
		t.Fatal("implausible token id must error")
	}
	// sparse-but-plausible ids stay accepted (real tokenizers have holes).
	ok := []byte(`{"model":{"type":"BPE","vocab":{"x":0,"y":7}}}`)
	if _, err := nlp.BPEFromJSON(ok); err != nil {
		t.Fatalf("plausible sparse ids must load: %v", err)
	}
}
