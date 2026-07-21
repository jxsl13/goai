package nlp

import (
	"math"
	"testing"
)

// BenchmarkNucleusTopP measures the per-call cost of top-p (nucleus) truncation,
// which runs once per generated token when top-p sampling is active. It allocated
// a fresh vocab-sized index slice (plus the radix-sort scratch) every call.
func BenchmarkNucleusTopP(b *testing.B) {
	const vocab = 32000
	probs := make([]float64, vocab)
	var sum float64
	for i := range probs {
		probs[i] = math.Exp(math.Sin(float64(i)*0.01) * 2) // a peaky-ish distribution
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	work := make([]float64, vocab)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		copy(work, probs) // fresh probs each call (nucleusTopP mutates in place)
		nucleusTopP(work, 0.9)
	}
}
