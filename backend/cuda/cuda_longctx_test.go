//go:build cuda && cgo && (linux || windows)

package cuda_test

import "testing"

// Decode throughput vs CONTEXT length: the Q8 fixed-buffer decode attends the full
// padded cache (maxSeq), so this measures whether long context (up to the model's
// 2048 ctx) degrades decode — or whether it stays weight-bandwidth-bound (weights
// ≫ KV reads/scores) and thus flat. Characterizes real-world usability.
func benchDecodeCtx(b *testing.B, ctx int) {
	m := loadTinyLlama(b)
	gd := buildQ8GraphDecoder(b, m, ctx+34)
	defer gd.free()
	for p := 0; p < ctx; p++ {
		gd.step(b, int32((p*131+1)%m.Config.Vocab), p)
	}
	tok := int32(gd.logits.Argmax())
	b.ResetTimer()
	for range b.N {
		for d := 0; d < 32; d++ {
			gd.step(b, tok, ctx+1+d)
			tok = int32(gd.logits.Argmax())
		}
	}
	b.ReportMetric(float64(32)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkTinyLlamaDecodeCtx_32(b *testing.B)   { benchDecodeCtx(b, 32) }
func BenchmarkTinyLlamaDecodeCtx_256(b *testing.B)  { benchDecodeCtx(b, 256) }
func BenchmarkTinyLlamaDecodeCtx_512(b *testing.B)  { benchDecodeCtx(b, 512) }
func BenchmarkTinyLlamaDecodeCtx_1024(b *testing.B) { benchDecodeCtx(b, 1024) }
