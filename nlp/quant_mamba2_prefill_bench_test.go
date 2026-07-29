package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// Two prefill benchmarks that look alike and measure different things. Both were written
// independently; keeping only one would silently drop the workload it covers.

// benchQuantMamba2PrefillQuant sweeps the QUANT TYPE at a small model size, because the
// multi-token path is the only workload that reaches QMatMul's m>1 general loop — every
// pre-existing quantized benchmark is single-token and takes a fused m==1 path instead.
// Verified by panic probe rather than assumed: a benchmark that never enters the loop it
// is meant to cover measures nothing and reads as evidence anyway.
func benchQuantMamba2PrefillQuant(b *testing.B, qt gguf.QuantType, seq int) {
	m := benchMamba2Model(256, 8, 32, 2, 16, 4, 2, 64)
	q, err := nlp.QuantizeMamba2(m, qt)
	if err != nil {
		b.Fatal(err)
	}
	defer q.Close()
	ctx := backend.NewContext()
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = (i * 7) % 64
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := q.Forward(ctx, tokens); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuantMamba2PrefillQ4_K_128(b *testing.B) {
	benchQuantMamba2PrefillQuant(b, gguf.Q4_K, 128)
}
func BenchmarkQuantMamba2PrefillQ8_0_128(b *testing.B) {
	benchQuantMamba2PrefillQuant(b, gguf.Q8_0, 128)
}

// benchQuantMamba2Prefill sweeps the SEQUENCE LENGTH at a production-ish geometry (64 heads,
// headDim 64), where the per-head SSD scan runs seq timesteps per head — the shape a prompt
// ingest actually hits, and the one the mixer parallelization was measured against.
func benchQuantMamba2Prefill(b *testing.B, seq int) {
	m := benchMamba2Model(512, 64, 64, 2, 16, 4, 2, 128)
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
