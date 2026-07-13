package nlp_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// a tiny tiktoken rank file: base64("a")=YQ==, base64("b")=Yg==, base64("ab")=YWI=.
const tiktokenSample = "YQ== 0\nYg== 1\nYWI= 2\n"

// §V15 / §V16-exempt (definitional format): TiktokenFromBytes parses the rank file into a working
// byte-level BPE.
func TestTiktokenParse(t *testing.T) {
	tok, err := nlp.TiktokenFromBytes([]byte(tiktokenSample))
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Decode([]int{2}); got != "ab" {
		t.Errorf("Decode([2]) = %q, want ab", got)
	}
	if got := tok.Decode([]int{0, 1}); got != "ab" {
		t.Errorf("Decode([0 1]) = %q, want ab", got)
	}
}

// §V15 (round-trip): TiktokenFromBytes(t.ToTiktoken()) reproduces the tokenizer — stable
// serialization (ascending by rank) and identical decoding.
func TestTiktokenRoundTrip(t *testing.T) {
	tok, err := nlp.TiktokenFromBytes([]byte(tiktokenSample))
	if err != nil {
		t.Fatal(err)
	}
	s1 := tok.ToTiktoken()
	tok2, err := nlp.TiktokenFromBytes(s1)
	if err != nil {
		t.Fatal(err)
	}
	if s2 := tok2.ToTiktoken(); !bytes.Equal(s1, s2) {
		t.Errorf("round-trip not stable:\n%q\n%q", s1, s2)
	}
	for _, id := range []int{0, 1, 2} {
		if tok.Decode([]int{id}) != tok2.Decode([]int{id}) {
			t.Errorf("decode of id %d differs after round-trip", id)
		}
	}
	// the serialized form is the canonical ascending-by-rank layout.
	if string(s1) != tiktokenSample {
		t.Errorf("ToTiktoken = %q, want %q", s1, tiktokenSample)
	}
}

func TestTiktokenErrors(t *testing.T) {
	if _, err := nlp.TiktokenFromBytes([]byte("@@@notbase64 0\n")); err == nil {
		t.Error("bad base64: want error")
	}
	if _, err := nlp.TiktokenFromBytes([]byte("YQ== notanint\n")); err == nil {
		t.Error("bad rank: want error")
	}
}

// §V15 fuzz: TiktokenFromBytes never panics on arbitrary bytes; any tokenizer it produces
// re-serializes stably.
func FuzzTiktokenRoundTrip(f *testing.F) {
	f.Add([]byte(tiktokenSample))
	f.Add([]byte("YQ== 0\n"))
	f.Add([]byte(""))
	f.Add([]byte("garbage\nlines\n"))
	f.Fuzz(func(t *testing.T, data []byte) {
		tok, err := nlp.TiktokenFromBytes(data)
		if err != nil {
			return // rejected — fine, as long as it did not panic
		}
		s1 := tok.ToTiktoken()
		tok2, err := nlp.TiktokenFromBytes(s1)
		if err != nil {
			t.Fatalf("re-parse of ToTiktoken failed: %v", err)
		}
		if s2 := tok2.ToTiktoken(); !bytes.Equal(s1, s2) {
			t.Fatalf("serialization not stable:\n%q\n%q", s1, s2)
		}
	})
}

func ExampleTiktokenFromBytes() {
	tok, _ := nlp.TiktokenFromBytes([]byte("YQ== 0\nYg== 1\nYWI= 2\n"))
	fmt.Println(tok.Decode([]int{2}))
	// Output:
	// ab
}
