package nn

import (
	"math"
	"testing"
)

// BenchmarkQuantizeNF4 covers NF4 block-quantization (QLoRA) — the per-element argmin
// codebook lookup (nearestNF4) is the dominant cost. 1M normally-distributed weights.
func BenchmarkQuantizeNF4(b *testing.B) {
	const n = 1 << 20
	w := make([]float64, n)
	// deterministic pseudo-normal spread across [-1,1] via a fixed LCG + tanh
	s := uint64(0x9e3779b97f4a7c15)
	for i := range w {
		s = s*6364136223846793005 + 1442695040888963407
		u := float64(s>>11) / float64(1<<53) // [0,1)
		w[i] = math.Tanh((u*2 - 1) * 2)      // concentrate near 0, tails to ±1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = QuantizeNF4(w, 64)
	}
}
