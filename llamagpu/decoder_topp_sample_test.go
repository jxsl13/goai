//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestDecoderTopPSampleMatchesFallback proves the generic Decoder's on-device PURE-top-p sampling
// fast-path (device TopKN(64) + SoftmaxStatsN + nlp.SampleTopPFromCandidates, host fallback on nucleus
// overflow) produces the EXACT same token sequence as the full-vocab CPU sampler fallback
// (GOAI_CUDA_TOPK_SAMPLE=0) for a penalty-free pure-top-p sampler with the same seed — the same
// soundness guarantee as the graph decoder's top-p fast-path (#1015), extended to the generic Decoder.
func TestDecoderTopPSampleMatchesFallback(t *testing.T) {
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
	// PURE top-p (no top-k) — the config that previously fell to the whole-vocab host path.
	mk := func() nlp.TokenSampler {
		return nlp.NewSampler(1234, nlp.WithTemperature(0.8), nlp.WithTopP(0.92))
	}
	decF, err := llamagpu.NewLlamaQ8CUDA(m)
	if err != nil {
		t.Fatal(err)
	}
	fast, err := decF.Generate(prompt, maxNew, mk())
	decF.Release()
	if err != nil {
		t.Fatalf("fast Generate: %v", err)
	}
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
			t.Fatalf("token %d: fast=%d fallback=%d — Decoder on-device pure-top-p diverges from full-vocab", i, fast[i], slow[i])
		}
	}
	t.Logf("Decoder on-device pure-top-p == full-vocab fallback over %d tokens (TopP=0.92, temp=0.8)", len(fast))
}
