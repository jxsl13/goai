package nlp

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func seedGGUFParts(mapped string) []ggufPart {
	var p []ggufPart
	for i := range mapped {
		p = append(p, ggufPart{start: i, rank: math.MaxInt})
	}
	return append(p, ggufPart{start: len(mapped), rank: math.MaxInt})
}

// TestBPEGGUFHeapMatchesQuad asserts the O(n log n) heap merge emits identical token ids to the O(n²)
// path for the GGUF BPETokenizer, across long letter pieces (h/e/l/o merge tree) + random sequences.
func TestBPEGGUFHeapMatchesQuad(t *testing.T) {
	tok, err := NewBPE(
		[]string{"h", "e", "l", "o", "he", "ll", "hello", "hell", "Ġ"},
		[]string{"h e", "l l", "he ll", "hell o"},
	)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(5, 11))
	cases := []string{strings.Repeat("hello", 40), strings.Repeat("hlelo", 60)}
	letters := []byte("hello")
	for c := 0; c < 200; c++ {
		n := 130 + rng.IntN(400)
		bs := make([]byte, n)
		for i := range bs {
			bs[i] = letters[rng.IntN(len(letters))]
		}
		cases = append(cases, string(bs))
	}
	for ci, mapped := range cases { // h/e/l/o map to identity under bytesToUnicode, so mapped == input
		if len(mapped) <= bpeHeapThreshold {
			continue
		}
		hp := tok.bpeIntoHeap(mapped, nil, seedGGUFParts(mapped))
		q := tok.bpeIntoQuad(mapped, nil, seedGGUFParts(mapped))
		if len(hp) != len(q) {
			t.Fatalf("case %d len %d: heap %d tokens vs quad %d", ci, len(mapped), len(hp), len(q))
		}
		for k := range hp {
			if hp[k] != q[k] {
				t.Fatalf("case %d: token %d heap=%d quad=%d", ci, k, hp[k], q[k])
			}
		}
	}
}

func BenchmarkBPEGGUFHeap(b *testing.B) {
	tok, _ := NewBPE(
		[]string{"h", "e", "l", "o", "he", "ll", "hello", "hell", "Ġ"},
		[]string{"h e", "l l", "he ll", "hell o"},
	)
	mapped := strings.Repeat("hello", 400) // 2000-char single piece
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.bpeIntoHeap(mapped, nil, seedGGUFParts(mapped))
	}
}
func BenchmarkBPEGGUFQuad(b *testing.B) {
	tok, _ := NewBPE(
		[]string{"h", "e", "l", "o", "he", "ll", "hello", "hell", "Ġ"},
		[]string{"h e", "l l", "he ll", "hell o"},
	)
	mapped := strings.Repeat("hello", 400)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.bpeIntoQuad(mapped, nil, seedGGUFParts(mapped))
	}
}
