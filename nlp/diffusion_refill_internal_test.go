package nlp

import (
	"math"
	"testing"
)

// BenchmarkDiffusionRefillOrder measures the per-step confidence sort that
// masked-diffusion generation runs each round. At 2048 masked positions the
// radix path (sortIdxAscByScore) is ~2.46× faster than the closure comparison
// sort (114µs -> 47µs); the ordering is identical for distinct confidences.
func BenchmarkDiffusionRefillOrder(b *testing.B) {
	const n = 2048
	confs := make([]float64, n)
	for i := range confs {
		confs[i] = math.Abs(math.Sin(float64(i) * 0.7))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = diffusionRefillOrder(confs)
	}
}
