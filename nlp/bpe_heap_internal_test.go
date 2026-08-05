package nlp

import (
	"math/rand/v2"
	"strings"
	"testing"
)

// TestBPEHeapMatchesQuadratic asserts the O(n log n) heap merge emits byte-for-byte the same token ids
// as the O(n²) rescan path, across random and adversarial large pieces (the tie-break and merge order
// must be identical).
func TestBPEHeapMatchesQuadratic(t *testing.T) {
	tk := loadInternalTok(t)
	rng := rand.New(rand.NewPCG(42, 7))
	cases := []string{
		strings.Repeat("1234567890", 500),
		strings.Repeat("+", 2000),
		strings.Repeat("=-_/", 1000),
		strings.Repeat(" ", 1000),
		"3141592653589793238462643383279502884197169399375105820974944592",
	}
	// random symbol/digit runs (single pieces after pre-tokenization)
	alpha := []byte("0123456789+-=_/.,:;!?@#$%^&*()")
	for c := 0; c < 200; c++ {
		n := 130 + rng.IntN(600)
		bs := make([]byte, n)
		for i := range bs {
			bs[i] = alpha[rng.IntN(len(alpha))]
		}
		cases = append(cases, string(bs))
	}
	for ci, s := range cases {
		p := []byte(s)
		if len(p) <= bpeHeapThreshold {
			continue
		}
		heapOut := tk.bpeMergeHeapInto(p, nil)
		// force the quadratic path by calling it directly
		var parts []bpePart
		quadOut, _ := tk.bpeMergeQuadInto(p, nil, parts)
		if len(heapOut) != len(quadOut) {
			t.Fatalf("case %d len %d: heap %d tokens vs quad %d", ci, len(p), len(heapOut), len(quadOut))
		}
		for k := range heapOut {
			if heapOut[k] != quadOut[k] {
				t.Fatalf("case %d len %d: token %d heap=%d quad=%d", ci, len(p), k, heapOut[k], quadOut[k])
			}
		}
	}
}

func benchBPEHeapPiece(b *testing.B, rep int) {
	tk := loadInternalTok(b)
	p := []byte(strings.Repeat("1234567890", rep))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.bpeMergeHeapInto(p, nil)
	}
}

// BenchmarkBPEMergeHeap_* track the O(n log n) large-piece merge (long digit/symbol runs); vs the
// quadratic path these are ~2.5x at 1000 bytes and ~7.6x at 4000, and the gap widens with length.
func BenchmarkBPEMergeHeap1000(b *testing.B) { benchBPEHeapPiece(b, 100) }
func BenchmarkBPEMergeHeap4000(b *testing.B) { benchBPEHeapPiece(b, 400) }
