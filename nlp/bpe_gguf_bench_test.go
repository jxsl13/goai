package nlp_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// BenchmarkBPEGGUFEncode drives the GGUF byte-level BPE Encode over a substantial
// text so the per-piece merge loop — the O(L²) candidate scan that built a "left
// right" merge-key string per pair per iteration (T625 class) — dominates.
func BenchmarkBPEGGUFEncode(b *testing.B) {
	vocab := []string{"Ġ"} // Ġ, the byte-mapped leading space
	for c := 'a'; c <= 'z'; c++ {
		vocab = append(vocab, string(c))
	}
	vocab = append(vocab, "th", "he", "the", "in", "ing", "er", "an", "on", "at", "en", "re", "ed", "nd", "ou", "ea", "st", "ar", "nt")
	merges := []string{"t h", "h e", "th e", "i n", "in g", "e r", "a n", "o n", "a t", "e n", "r e", "e d", "n d", "o u", "e a", "s t", "a r", "n t"}
	tok, err := nlp.NewBPE(vocab, merges)
	if err != nil {
		b.Fatal(err)
	}
	text := strings.Repeat("the rain in spain remained on the ground and then rested ", 2000)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = tok.Encode(text)
	}
}
