package nlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

func peakedLogits(vocab int) []float64 {
	r := rand.New(rand.NewPCG(1, 2))
	l := make([]float64, vocab)
	for i := range l {
		l[i] = -12 + 2*r.Float64()
	}
	for i := 0; i < 40; i++ {
		l[r.IntN(vocab)] = 4 + 6*r.Float64()
	}
	return l
}
func flatLogits(vocab int) []float64 {
	l := make([]float64, vocab)
	for i := range l {
		l[i] = math.Sin(float64(i) * 0.001)
	}
	return l
}
func BenchmarkDistPeakedTopP(b *testing.B) {
	l := peakedLogits(128000)
	s := NewSampler(42, WithTemperature(0.8), WithTopP(0.95))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Dist(l)
	}
}
func BenchmarkDistPeakedTopK(b *testing.B) {
	l := peakedLogits(128000)
	s := NewSampler(42, WithTemperature(0.8), WithTopK(50))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Dist(l)
	}
}
