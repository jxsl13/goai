package nlp_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

func benchUnigram() (*nlp.Unigram, string) {
	var vocab []nlp.UnigramPiece
	vocab = append(vocab, nlp.UnigramPiece{Piece: "<unk>", Score: 0})
	vocab = append(vocab, nlp.UnigramPiece{Piece: "▁", Score: -3})
	for c := 'a'; c <= 'z'; c++ {
		vocab = append(vocab, nlp.UnigramPiece{Piece: string(c), Score: -6})
	}
	// common multi-char pieces (give the DP real inner-loop work)
	for _, p := range []string{"th", "the", "▁the", "in", "ing", "er", "▁a", "▁to", "on", "at", "en", "▁and", "or", "es", "▁of", "ion", "tion"} {
		vocab = append(vocab, nlp.UnigramPiece{Piece: p, Score: -5})
	}
	u, err := nlp.NewUnigram(vocab)
	if err != nil {
		panic(err)
	}
	text := strings.Repeat("the cat sat on the mat in the barn and ate the corn ", 40)
	return u, text
}

func BenchmarkUnigramEncode(b *testing.B) {
	u, text := benchUnigram()
	b.ReportAllocs()
	for b.Loop() {
		_ = u.Encode(text)
	}
}

func BenchmarkSPMEncode(b *testing.B) {
	var vocab []nlp.UnigramPiece
	vocab = append(vocab, nlp.UnigramPiece{Piece: "<unk>", Score: 0})
	vocab = append(vocab, nlp.UnigramPiece{Piece: "▁", Score: 0})
	for c := 'a'; c <= 'z'; c++ {
		vocab = append(vocab, nlp.UnigramPiece{Piece: string(c), Score: 0})
	}
	// merge pieces with descending (negative) merge ranks
	for i, p := range []string{"th", "the", "▁the", "in", "ing", "er", "▁a", "▁to", "on", "at", "en", "▁and", "or", "es", "▁of", "ion", "tion", "▁cat", "▁sat", "▁mat"} {
		vocab = append(vocab, nlp.UnigramPiece{Piece: p, Score: -float64(i)})
	}
	u, err := nlp.NewSPM(vocab)
	if err != nil {
		panic(err)
	}
	text := strings.Repeat("the cat sat on the mat in the barn and ate the corn ", 20)
	b.ReportAllocs()
	for b.Loop() {
		_ = u.Encode(text)
	}
}
