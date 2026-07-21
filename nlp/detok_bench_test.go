package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// unigramBenchVocab is the small SPM/Unigram vocab shared by the encode benchmarks:
// <unk>, the ▁ space-meta, a–z, then 20 multi-char merges. Ids 2..47 are real pieces.
func unigramBenchVocab() []nlp.UnigramPiece {
	v := []nlp.UnigramPiece{{Piece: "<unk>", Score: 0}, {Piece: "▁", Score: 0}}
	for c := 'a'; c <= 'z'; c++ {
		v = append(v, nlp.UnigramPiece{Piece: string(c), Score: 0})
	}
	for i, p := range []string{"th", "the", "▁the", "in", "ing", "er", "▁a", "▁to", "on", "at", "en", "▁and", "or", "es", "▁of", "ion", "tion", "▁cat", "▁sat", "▁mat"} {
		v = append(v, nlp.UnigramPiece{Piece: p, Score: -float64(i)})
	}
	return v
}

// cycleBenchIDs returns n ids cycling through the real-piece range [lo,hi) so Decode
// reconstructs a large (~sub-MB) string — the regime where the strings.Builder growth
// churn (T929) shows up over the tiny per-id cost.
func cycleBenchIDs(n, lo, hi int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = lo + i%(hi-lo)
	}
	return ids
}

func BenchmarkSPMDecode(b *testing.B) {
	s, err := nlp.NewSPM(unigramBenchVocab())
	if err != nil {
		b.Fatal(err)
	}
	ids := cycleBenchIDs(300000, 2, 48)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = s.Decode(ids)
	}
}

func BenchmarkUnigramDecode(b *testing.B) {
	u, err := nlp.NewUnigram(unigramBenchVocab())
	if err != nil {
		b.Fatal(err)
	}
	ids := cycleBenchIDs(300000, 2, 48)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = u.Decode(ids)
	}
}

func BenchmarkWordPieceDecode(b *testing.B) {
	v := []string{"[UNK]", "[CLS]", "[SEP]"}
	for c := 'a'; c <= 'z'; c++ {
		v = append(v, string(c))
	}
	v = append(v, "the", "cat", "sat", "on", "mat", "##ing", "##ed", "##s", "##er", "##ion")
	w, err := nlp.NewWordPiece(v)
	if err != nil {
		b.Fatal(err)
	}
	ids := cycleBenchIDs(300000, 3, len(v))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = w.Decode(ids)
	}
}
