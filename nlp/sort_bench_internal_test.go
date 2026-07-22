package nlp

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// BenchmarkJLensReadout ranks a full [1, vocab] logit row — the large-n sort at the
// heart of the logit lens. A pseudo-random (deterministic, LCG) fill defeats any
// nearly-sorted fast path so the benchmark reflects a real logit distribution.
func BenchmarkJLensReadout(b *testing.B) {
	const vocab = 128_000
	logits := tensor.New(tensor.F64, tensor.Shape{1, vocab})
	s := logits.Storage().F64()
	x := uint64(0x9e3779b97f4a7c15)
	for i := range s {
		x = x*6364136223846793005 + 1442695040888963407
		s[i] = float64(int64(x>>11)) / (1 << 53) // deterministic [0,1)
	}
	b.ResetTimer()
	for range b.N {
		_ = newJLensReadout(logits, 0)
	}
}

// BenchmarkCosineRerank reranks a large candidate set against a query — the sort is
// over every candidate, so it dominates once the corpus is large.
func BenchmarkCosineRerank(b *testing.B) {
	const n, dim = 20_000, 64
	query := make([]float64, dim)
	cands := make([][]float64, n)
	x := uint64(0xdeadbeefcafef00d)
	nextF := func() float64 {
		x = x*6364136223846793005 + 1442695040888963407
		return float64(int64(x>>11))/(1<<53) - 0.5
	}
	for j := range query {
		query[j] = nextF()
	}
	for i := range cands {
		c := make([]float64, dim)
		for j := range c {
			c[j] = nextF()
		}
		cands[i] = c
	}
	b.ResetTimer()
	for range b.N {
		if _, err := CosineRerank(query, cands); err != nil {
			b.Fatal(err)
		}
	}
}
