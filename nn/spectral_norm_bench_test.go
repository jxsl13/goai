package nn

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSpectralNormPowerIterate covers the power-iteration spectral-norm estimate (two
// matvecs per round) refreshed every forward. in=512, out=512, 10 rounds.
func BenchmarkSpectralNormPowerIterate(b *testing.B) {
	s := NewSpectralNorm(tensor.F64, 512, 512, 42)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		s.powerIterate(10)
	}
}
