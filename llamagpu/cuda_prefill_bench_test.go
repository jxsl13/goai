//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// prefillLlama is realLlama's geometry (Dim=2048/8L/Hidden=5632, GQA 16:4) but with a context long
// enough to prefill a ≥128-token prompt in one batched StepN — the regime where the recorder routes
// the quant projections to the tensor-core prefill GEMM (q4kWMMAThreshold=128). This is the e2e
// counterpart to the per-GEMM BenchmarkQ4KPrefill_* in backend/cuda: it measures whole-model
// prompt-processing (prefill) throughput, the metric that maps to llama.cpp's `pp` / "prompt eval".
func prefillLlama(tb testing.TB, ctx int) *nlp.Llama {
	tb.Helper()
	cfg := nlp.LlamaConfig{
		Vocab: 32000, Ctx: ctx, Dim: 2048, Heads: 16, KVHeads: 4, Layers: 8,
		Hidden: 5632, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

// benchLlamaPrefillN times a batched StepN prefill of `seq` tokens from position 0 (StepN resets the
// cache at pos==0, so each iteration is an independent prefill) and reports whole-model prefill
// throughput in tok/s.
func benchLlamaPrefillN(b *testing.B, dec *llamagpu.Decoder, seq int) {
	b.Helper()
	prompt := make([]int, seq)
	for i := range prompt {
		prompt[i] = (i*7)%31991 + 1
	}
	if _, err := dec.StepN(prompt, 0); err != nil { // prime (warms scratch + graph)
		b.Fatalf("prime StepN: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dec.StepN(prompt, 0); err != nil {
			b.Fatalf("StepN: %v", err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkLlamaPrefillF32_seq512(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewCUDA(prefillLlama(b, 512))
	if err != nil {
		b.Fatalf("NewCUDA: %v", err)
	}
	defer dec.Release()
	benchLlamaPrefillN(b, dec, 512)
}

func BenchmarkLlamaPrefillQ8_seq512(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewLlamaQ8CUDA(prefillLlama(b, 512))
	if err != nil {
		b.Fatalf("NewLlamaQ8CUDA: %v", err)
	}
	defer dec.Release()
	benchLlamaPrefillN(b, dec, 512)
}

func BenchmarkLlamaPrefillQ4K_seq512(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewLlamaQ4KCUDA(prefillLlama(b, 512))
	if err != nil {
		b.Fatalf("NewLlamaQ4KCUDA: %v", err)
	}
	defer dec.Release()
	benchLlamaPrefillN(b, dec, 512)
}

// tinyLlamaGeom is TinyLlama-1.1B-Chat's exact architecture (Dim=2048, 22 layers, 32 heads / 4 KV
// heads GQA, Hidden=5632) — throughput is a function of shape not weight values, so a synthetic model
// at this geometry gives a whole-model prefill tok/s directly comparable to llama.cpp's `pp512` on the
// real tinyllama-1.1b-chat GGUF (measured 9047 tok/s, b10259 Vulkan on this RTX 3060).
func tinyLlamaGeom(tb testing.TB, ctx int) *nlp.Llama {
	tb.Helper()
	cfg := nlp.LlamaConfig{
		Vocab: 32000, Ctx: ctx, Dim: 2048, Heads: 32, KVHeads: 4, Layers: 22,
		Hidden: 5632, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

func BenchmarkLlamaPrefillQ4K_tinyllama_seq512(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewLlamaQ4KCUDA(tinyLlamaGeom(b, 512))
	if err != nil {
		b.Fatalf("NewLlamaQ4KCUDA: %v", err)
	}
	defer dec.Release()
	benchLlamaPrefillN(b, dec, 512)
}
