package nn

import (
	"math"
	"testing"
)

func BenchmarkHQQuantize(b *testing.B) {
	const n = 2048 * 2048
	w := make([]float64, n)
	s := uint64(0x243f6a8885a308d3)
	for i := range w {
		s = s*6364136223846793005 + 1442695040888963407
		u := float64(s>>11) / float64(1<<53)
		w[i] = math.Tanh((u*2 - 1) * 2)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _, _ = HQQuantize(w, 4, 64)
	}
}
