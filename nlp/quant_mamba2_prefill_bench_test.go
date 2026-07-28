package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

<<<<<<< HEAD
// benchQuantMamba2Prefill measures the multi-token prefill mixer (QuantMamba2Mixer.forward),
// where the per-head SSD scan runs seq timesteps per head — the workload a prompt ingest hits.
func benchQuantMamba2Prefill(b *testing.B, seq int) {
	m := benchMamba2Model(512, 64, 64, 2, 16, 4, 2, 128) // 64 heads, headDim 64 — production-ish
	q, err := nlp.QuantizeMamba2(m, gguf.Q8_0)
=======
// benchQuantMamba2Prefill measures the MULTI-token (prefill) path, which is the only
// workload that reaches QMatMul's m>1 general loop. Every pre-existing quantized
// benchmark is single-token and takes a fused m==1 path instead — verified by panic
// probe, not assumed, because a benchmark that never enters the loop it is meant to
// cover measures nothing and reads as evidence anyway.
func benchQuantMamba2Prefill(b *testing.B, qt gguf.QuantType, seq int) {
	m := benchMamba2Model(256, 8, 32, 2, 16, 4, 2, 64)
	q, err := nlp.QuantizeMamba2(m, qt)
>>>>>>> 3855e4bf (perf(format/gguf): parallelize the m>1 QMatMul weight-row loop — quantized prefill 2.52x)
	if err != nil {
		b.Fatal(err)
	}
	defer q.Close()
<<<<<<< HEAD
	ctx := backend.NewContext()
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = (i * 7) % 128
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := q.Forward(ctx, tokens); err != nil {
=======
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = (i * 7) % 64
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := q.Forward(backend.NewContext(), tokens); err != nil {
>>>>>>> 3855e4bf (perf(format/gguf): parallelize the m>1 QMatMul weight-row loop — quantized prefill 2.52x)
			b.Fatal(err)
		}
	}
}

<<<<<<< HEAD
func BenchmarkQuantMamba2Prefill_256(b *testing.B) { benchQuantMamba2Prefill(b, 256) }
func BenchmarkQuantMamba2Prefill_512(b *testing.B) { benchQuantMamba2Prefill(b, 512) }
=======
func BenchmarkQuantMamba2PrefillQ4_K_128(b *testing.B) {
	benchQuantMamba2Prefill(b, gguf.Q4_K, 128)
}
func BenchmarkQuantMamba2PrefillQ8_0_128(b *testing.B) {
	benchQuantMamba2Prefill(b, gguf.Q8_0, 128)
}
>>>>>>> 3855e4bf (perf(format/gguf): parallelize the m>1 QMatMul weight-row loop — quantized prefill 2.52x)
