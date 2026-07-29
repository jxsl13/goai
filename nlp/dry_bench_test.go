package nlp

import "testing"

// BenchmarkApplyDRY covers the DRY repetition penalty over a full context window with
// realistic repetition (which drives the O(L²) suffix scan), vocab 32000, DRYRange 2048.
func BenchmarkApplyDRY(b *testing.B) {
	const vocab, L = 32000, 2048
	logits := make([]float64, vocab)
	for i := range logits {
		logits[i] = float64((i % 251)) * 0.01
	}
	// window with short-period repetition so the inner suffix loop runs deep.
	hist := make([]int, L)
	for i := range hist {
		hist[i] = (i % 37) + 5
	}
	s := &Sampler{
		DRYMultiplier: 0.8,
		DRYBase:       1.75,
		DRYAllowedLen: 2,
		DRYRange:      L,
		DRYBreakers:   []int{0, 1, 2, 10, 13},
	}
	work := make([]float64, vocab)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		copy(work, logits) // applyDRY mutates in place
		s.applyDRY(work, hist)
	}
}
