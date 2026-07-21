package nlp

import "testing"

// BenchmarkApplyPenalties measures the per-call cost with a realistic late-decode
// history (the map[int]int is rebuilt every call over the whole history).
func BenchmarkApplyPenalties(b *testing.B) {
	const vocab, hist = 32000, 2048
	logits := make([]float64, vocab)
	for i := range logits {
		logits[i] = float64(i%7) - 3
	}
	generated := make([]int, hist)
	for i := range generated {
		generated[i] = (i * 131) % vocab // spread across the vocab, some repeats
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		ApplyPenalties(logits, generated, 1.3, 0.5, 0.2)
	}
}
