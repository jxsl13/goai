//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestDecoderTopKSampleMatchesFallback proves the generic Decoder's on-device top-k sampling fast-path
// (Q8 + all families, via the deviceTopKer interface on the CUDA logits buffer) produces the EXACT same
// token sequence as the full-vocab CPU sampler fallback (GOAI_CUDA_TOPK_SAMPLE=0) for a penalty-free
// TopK sampler with the same seed — the same soundness guarantee as the graph decoder's fast-path.
func TestDecoderTopKSampleMatchesFallback(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 2048, Ctx: 128, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 9, 42, 17}
	const maxNew = 48
	mk := func() nlp.TokenSampler {
		return nlp.NewSampler(1234, nlp.WithTemperature(0.8), nlp.WithTopK(40), nlp.WithTopP(0.92))
	}
	// fast-path (default)
	decF, err := llamagpu.NewLlamaQ8CUDA(m)
	if err != nil {
		t.Fatal(err)
	}
	fast, err := decF.Generate(prompt, maxNew, mk())
	decF.Release()
	if err != nil {
		t.Fatalf("fast Generate: %v", err)
	}
	// forced fallback
	t.Setenv("GOAI_CUDA_TOPK_SAMPLE", "0")
	decS, err := llamagpu.NewLlamaQ8CUDA(m)
	if err != nil {
		t.Fatal(err)
	}
	slow, err := decS.Generate(prompt, maxNew, mk())
	decS.Release()
	if err != nil {
		t.Fatalf("fallback Generate: %v", err)
	}
	if len(fast) != len(slow) {
		t.Fatalf("length mismatch: fast %d, fallback %d", len(fast), len(slow))
	}
	for i := range fast {
		if fast[i] != slow[i] {
			t.Fatalf("token %d: fast=%d fallback=%d — Decoder on-device top-k sampling diverges from full-vocab", i, fast[i], slow[i])
		}
	}
	t.Logf("Decoder on-device top-k sampling == full-vocab fallback over %d tokens", len(fast))
}

func benchDecoderGenerate(b *testing.B, vocab, maxNew int) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: vocab, Ctx: 128, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		b.Fatal(err)
	}
	dec, err := llamagpu.NewLlamaQ8CUDA(m)
	if err != nil {
		b.Fatal(err)
	}
	defer dec.Release()
	prompt := []int{1, 2, 3, 4}
	mk := func() nlp.TokenSampler {
		return nlp.NewSampler(1, nlp.WithTemperature(0.8), nlp.WithTopK(40), nlp.WithTopP(0.9))
	}
	if _, err := dec.Generate(prompt, 2, mk()); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dec.Generate(prompt, maxNew, mk()); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(maxNew)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkDecoderGenerate128k(b *testing.B) { benchDecoderGenerate(b, 128000, 64) }

// benchDecoderPrefill isolates the prefill LM-head cost: a long prompt with maxNew=1 at a LARGE vocab,
// where computing/downloading the logits of every prompt row (vs only the last) is the dominant waste.
func benchDecoderPrefill(b *testing.B, vocab, promptLen int) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: vocab, Ctx: promptLen + 8, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		b.Fatal(err)
	}
	dec, err := llamagpu.NewLlamaQ8CUDA(m)
	if err != nil {
		b.Fatal(err)
	}
	defer dec.Release()
	prompt := make([]int, promptLen)
	for i := range prompt {
		prompt[i] = (i*7 + 1) % vocab
	}
	mk := func() nlp.TokenSampler { return nlp.Greedy() }
	if _, err := dec.Generate(prompt, 1, mk()); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dec.Generate(prompt, 1, mk()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecoderPrefill128k_512(b *testing.B) { benchDecoderPrefill(b, 128000, 512) }
