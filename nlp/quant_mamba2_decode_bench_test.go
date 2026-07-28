package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// benchQuantMamba2Decode measures the single-token recurrent step — the workload
// generation actually runs. It reuses benchMamba2Model and quantizes it, so the
// projections are the SIMD block-dot path a real checkpoint uses; the SSD scan around
// them stays scalar float64, which is why its share of the step is larger than its
// share of the arithmetic.
func benchQuantMamba2Decode(b *testing.B, qt gguf.QuantType) {
	m := benchMamba2Model(256, 8, 32, 2, 16, 4, 2, 64)
	q, err := nlp.QuantizeMamba2(m, qt)
	if err != nil {
		b.Fatal(err)
	}
	defer q.Close()
	ctx := backend.NewContext()
	st := q.NewDecodeState()
	// Warm the state so the conv window is full rather than zero-padded.
	for i := range 8 {
		if _, err := q.DecodeStep(ctx, st, i%64); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := q.DecodeStep(ctx, st, (i*7)%64); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuantMamba2DecodeQ8_0(b *testing.B) { benchQuantMamba2Decode(b, gguf.Q8_0) }
func BenchmarkQuantMamba2DecodeQ4_0(b *testing.B) { benchQuantMamba2Decode(b, gguf.Q4_0) }

// The K-quants are what PS6003 still reports as uncovered by a fused single-token path.
// Q4_K and Q6_K are llama.cpp's common deployment formats, so they carry the leverage.
func BenchmarkQuantMamba2DecodeQ4_K(b *testing.B) { benchQuantMamba2Decode(b, gguf.Q4_K) }
func BenchmarkQuantMamba2DecodeQ6_K(b *testing.B) { benchQuantMamba2Decode(b, gguf.Q6_K) }
