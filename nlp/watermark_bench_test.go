package nlp_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// Committed A/B harness for the watermark green-list hot paths (§V22/§V38), so the
// Detect/BiasLogits win is re-measurable and cannot rot. Detect over a realistic
// vocab×sequence is the headline: it scores one green membership per token, and the
// pre-optimization form minted a fresh VocabSize perm+mask per token (O(T·|V|)
// allocation, GC-bound). Both paths now reuse one identity permutation.

const (
	benchWMVocab = 32000
	benchWMSeq   = 1024
)

func BenchmarkWatermarkDetect(b *testing.B) {
	w, err := nlp.NewWatermark(benchWMVocab)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	toks := make([]int, benchWMSeq)
	for i := range toks {
		toks[i] = rng.IntN(benchWMVocab)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, s := w.Detect(toks); s != benchWMSeq-1 {
			b.Fatalf("scored %d", s)
		}
	}
}

func BenchmarkWatermarkBiasLogits(b *testing.B) {
	w, err := nlp.NewWatermark(benchWMVocab)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	logits := make([]float64, benchWMVocab)
	for i := range logits {
		logits[i] = rng.NormFloat64()
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := w.BiasLogits(logits, 7); err != nil {
			b.Fatal(err)
		}
	}
}

// TestWatermarkGreenPathsMatchMask pins the optimized Detect and BiasLogits to the
// unchanged public GreenMask reference across randomized vocab sizes, γ, δ and keys —
// the green set each derives MUST be the identical membership GreenMask reports, so
// the perm-reuse rewrite is bit-exact (§V16 tier-1 equivalence, guarding the §V22
// bit-identical requirement against future edits to the green-list machinery).
func TestWatermarkGreenPathsMatchMask(t *testing.T) {
	seed := rand.New(rand.NewPCG(99, 100))
	for trial := 0; trial < 300; trial++ {
		vocab := 2 + seed.IntN(600)
		w, err := nlp.NewWatermark(vocab,
			nlp.WithWatermarkGamma(0.05+0.9*seed.Float64()),
			nlp.WithWatermarkDelta(seed.NormFloat64()*3),
			nlp.WithWatermarkKey(seed.Uint64()))
		if err != nil {
			t.Fatal(err)
		}
		// BiasLogits must equal applying δ exactly where GreenMask is green — bit-exact.
		logits := make([]float64, vocab)
		for i := range logits {
			logits[i] = seed.NormFloat64()
		}
		prev := seed.IntN(vocab)
		got, err := w.BiasLogits(logits, prev)
		if err != nil {
			t.Fatal(err)
		}
		mask := w.GreenMask(prev)
		for i := range logits {
			want := logits[i]
			if mask[i] {
				want = logits[i] + w.Delta
			}
			if got[i] != want {
				t.Fatalf("trial %d vocab %d BiasLogits[%d]=%v want %v", trial, vocab, i, got[i], want)
			}
		}
		// Detect's green count must equal an independent GreenMask recount.
		seqLen := 1 + seed.IntN(60)
		toks := make([]int, seqLen)
		for i := range toks {
			toks[i] = seed.IntN(vocab)
		}
		_, green, scored := w.Detect(toks)
		var refGreen int
		for i := 1; i < len(toks); i++ {
			if w.GreenMask(toks[i-1])[toks[i]] {
				refGreen++
			}
		}
		if green != refGreen || scored != len(toks)-1 {
			t.Fatalf("trial %d vocab %d Detect green=%d scored=%d want green=%d scored=%d",
				trial, vocab, green, scored, refGreen, len(toks)-1)
		}
	}
}
