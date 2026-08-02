package nlp

import (
	"testing"

	"github.com/jxsl13/goai/format/gguf"
)

// BenchmarkQuantLlamaGenerate500 measures Q8_0 quantized KV-cached decode over the
// same geometry as BenchmarkLlamaGenerate500RowBuf (dim 256, 4 layers), so the
// quantized decode path — the way most downloaded community models actually run —
// has a permanent §V22 baseline next to the float one. The quantized projections go
// through nn.QuantLinear (ggml QMatMul / the GPU kernels); everything else (RoPE,
// attention, the KV cache) is shared with the float path.
func BenchmarkQuantLlamaGenerate500(b *testing.B) {
	m := kvBenchLlama(b)
	q, err := QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		b.Fatalf("QuantizeLlama: %v", err)
	}
	defer q.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		out, err := q.Generate([]int{1}, 500, Greedy())
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		if len(out) != 501 {
			b.Fatalf("generated %d tokens, want 501", len(out))
		}
	}
}
