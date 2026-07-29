package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// benchQuantMamba2Prefill measures the multi-token prefill mixer (QuantMamba2Mixer.forward),
// where the per-head SSD scan runs seq timesteps per head — the workload a prompt ingest hits.
func benchQuantMamba2Prefill(b *testing.B, seq int) {
	m := benchMamba2Model(512, 64, 64, 2, 16, 4, 2, 128) // 64 heads, headDim 64 — production-ish
	q, err := nlp.QuantizeMamba2(m, gguf.Q8_0)
	if err != nil {
		b.Fatal(err)
	}
	defer q.Close()
	ctx := backend.NewContext()
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = (i * 7) % 128
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := q.Forward(ctx, tokens); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuantMamba2Prefill_256(b *testing.B) { benchQuantMamba2Prefill(b, 256) }
func BenchmarkQuantMamba2Prefill_512(b *testing.B) { benchQuantMamba2Prefill(b, 512) }
