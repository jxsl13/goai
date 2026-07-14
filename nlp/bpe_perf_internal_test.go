package nlp

import (
	"math"
	"strings"
	"testing"
)

// bpeMergeNaive is the pre-optimization O(L²)-with-allocations merge (a candidate
// pair re-concatenated two strings per scan step). Kept HERE, in the test binary
// only, as the A/B baseline for BenchmarkBPEMerge* and as the reference for
// TestBPEMergeNaiveParity — the same-session A/B discipline (§V22), no git stash.
func bpeMergeNaive(t *Tokenizer, piece []byte) []int {
	if len(piece) == 1 {
		return []int{t.ranks[string(piece)]}
	}
	parts := make([][]byte, len(piece))
	for i := range piece {
		parts[i] = piece[i : i+1]
	}
	for len(parts) > 1 {
		minRank, minI := math.MaxInt, -1
		for i := 0; i < len(parts)-1; i++ {
			if r, ok := t.ranks[string(parts[i])+string(parts[i+1])]; ok && r < minRank {
				minRank, minI = r, i
			}
		}
		if minI < 0 {
			break
		}
		merged := append(append([]byte{}, parts[minI]...), parts[minI+1]...)
		parts[minI] = merged
		parts = append(parts[:minI+1], parts[minI+2:]...)
	}
	ids := make([]int, len(parts))
	for i, p := range parts {
		ids[i] = t.ranks[string(p)]
	}
	return ids
}

// bpePerfCorpus is a mix of natural language, code-ish identifiers, long digit
// runs and non-Latin script — the classes where BPE pieces get long enough that
// the per-pair allocations of the naive scan dominate.
const bpePerfCorpus = `The quick brown fox jumps over the lazy dog. ` +
	`Byte-pair encoding merges the most frequent adjacent symbol pair repeatedly. ` +
	`func (t *Tokenizer) bpeMerge(piece []byte) []int { return nil } // internalIdentifierNameThatIsQuiteLong ` +
	`0123456789012345678901234567890123456789 antidisestablishmentarianism supercalifragilisticexpialidocious ` +
	`https://example.com/very/long/path/segment?query=parameter&another=value#fragment-identifier ` +
	`Ünïcödé and 漢字と日本語のテキスト and mixed	whitespace   runs and emoji 🎉🎊✨ end.`

func loadInternalTok(tb testing.TB) *Tokenizer {
	tb.Helper()
	tk, err := LoadGPT2("testdata/gpt2_ranks.txt")
	if err != nil {
		tb.Skipf("no gpt2 ranks (run make golden): %v", err)
	}
	return tk
}

// TestBPEMergeNaiveParity locks the optimized merge to the old algorithm piece
// for piece across the corpus — the leftmost-lowest-rank order and the final
// boundaries must be bit-identical (this is what keeps the tiktoken golden green).
func TestBPEMergeNaiveParity(t *testing.T) {
	tk := loadInternalTok(t)
	for _, piece := range gpt2Split(bpePerfCorpus) {
		pb := []byte(piece)
		got := tk.bpeMerge(pb)
		want := bpeMergeNaive(tk, pb)
		if len(got) != len(want) {
			t.Fatalf("piece %q: len %d != naive %d (%v vs %v)", piece, len(got), len(want), got, want)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("piece %q [%d]: %d != naive %d", piece, i, got[i], want[i])
			}
		}
	}
}

func benchBPEPieces(tb testing.TB) [][]byte {
	tb.Helper()
	// a few corpus repeats so the benchmark body does real per-op work
	sp := gpt2Split(strings.Repeat(bpePerfCorpus+" ", 4))
	out := make([][]byte, len(sp))
	for i, s := range sp {
		out[i] = []byte(s)
	}
	return out
}

func BenchmarkBPEMergeNew(b *testing.B) {
	tk := loadInternalTok(b)
	pieces := benchBPEPieces(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range pieces {
			_ = tk.bpeMerge(p)
		}
	}
}

func BenchmarkBPEMergeNaive(b *testing.B) {
	tk := loadInternalTok(b)
	pieces := benchBPEPieces(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range pieces {
			_ = bpeMergeNaive(tk, p)
		}
	}
}
