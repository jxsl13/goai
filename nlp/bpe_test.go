package nlp_test

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

func loadTokenizer(t *testing.T) *nlp.Tokenizer {
	t.Helper()
	tk, err := nlp.LoadGPT2("testdata/gpt2_ranks.txt")
	if err != nil {
		t.Fatalf("load gpt2 ranks (run make golden): %v", err)
	}
	return tk
}

type tokGolden struct {
	Samples []struct {
		Text string `json:"text"`
		IDs  []int  `json:"ids"`
	} `json:"samples"`
}

// §V1: our byte-level BPE encoding is bit-identical to real tiktoken gpt2 across
// varied text (ascii, unicode, digits, punctuation, contractions, whitespace).
func TestBPEEncodeParityTiktoken(t *testing.T) {
	tk := loadTokenizer(t)
	raw, err := os.ReadFile("testdata/tokenizer.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g tokGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	for _, s := range g.Samples {
		got := tk.Encode(s.Text)
		if len(got) != len(s.IDs) {
			t.Errorf("%q: got %v, want %v (tiktoken)", s.Text, got, s.IDs)
			continue
		}
		for i := range got {
			if got[i] != s.IDs[i] {
				t.Errorf("%q [%d]: got %d want %d (tiktoken)", s.Text, i, got[i], s.IDs[i])
				break
			}
		}
	}
}

// §V15: decode(encode(x)) == x for arbitrary text (byte-level base vocab).
func TestBPERoundTrip(t *testing.T) {
	tk := loadTokenizer(t)
	cases := []string{
		"Hello, world!", "", "   ", "\t\n mixed\twhitespace\n",
		"Ünïcödé and 漢字 and 🎉 emoji", "1234567890", "a'b'c don't",
		"MixedCASE_and-symbols@#$%^&*()",
	}
	for _, s := range cases {
		if got := tk.Decode(tk.Encode(s)); got != s {
			t.Errorf("round-trip failed: %q → %q", s, got)
		}
	}
}

// §V15: fuzz — decode(encode(x)) must reconstruct any UTF-8 input exactly.
func FuzzBPERoundTrip(f *testing.F) {
	tk, err := nlp.LoadGPT2("testdata/gpt2_ranks.txt")
	if err != nil {
		f.Skip("no ranks file")
	}
	f.Add("Hello world")
	f.Add("  spaces  ")
	f.Add("Ünïcödé 漢字")
	f.Fuzz(func(t *testing.T, s string) {
		got := tk.Decode(tk.Encode(s))
		if got != s {
			t.Fatalf("round-trip: %q → %q", s, got)
		}
	})
}
