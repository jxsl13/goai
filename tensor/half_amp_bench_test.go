package tensor

import (
	"math"
	"testing"
)

// BenchmarkRoundToHalfF32Varied exercises the AMP master→compute sync hot path
// (RoundToHalfF32, F16 branch → f32ToF16 per element) over varied normal-range
// f32 with a random 23-bit mantissa, so the RNE round bit is ~50/50 — the real
// mixed-precision-training distribution. The existing BenchmarkF32ToF16 feeds
// float32(i)*0.25 (low mantissa bits ~always 0), which hides the rounding cost;
// this one measures it. Deterministic xorshift32 (no math/rand) → stable in CI.
func BenchmarkRoundToHalfF32Varied(b *testing.B) {
	const n = 262144
	src := make([]float32, n)
	dst := make([]float32, n)
	x := uint32(0x9e3779b9)
	for i := range src {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		// normal-range f32: biased exponent in [112,143], random 23-bit mantissa.
		bits := (uint32(112+(x&31)) << 23) | (x & 0x7FFFFF)
		src[i] = math.Float32frombits(bits)
	}
	b.ResetTimer()
	for range b.N {
		RoundToHalfF32(dst, src, F16)
	}
}
