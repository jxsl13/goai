package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
)

// benchMamba2Forward measures the full-sequence Mamba2.Forward (the float twin of
// QuantMamba2.Forward), whose per-layer Mixer.forward runs the scalar SSD scan / conv /
// gated-RMSNorm over all seq timesteps — the loops parallelized here.
func benchMamba2Forward(b *testing.B, seq int) {
	m := benchMamba2Model(512, 64, 64, 2, 16, 4, 2, 128) // 64 heads, headDim 64 — production-ish
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = (i * 7) % 128
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := m.Forward(ctx, tokens); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMamba2Forward_256(b *testing.B) { benchMamba2Forward(b, 256) }
func BenchmarkMamba2Forward_512(b *testing.B) { benchMamba2Forward(b, 512) }
