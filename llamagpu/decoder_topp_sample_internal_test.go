//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
)

// TestGraphLlamaTopPSampleMatchesFallback proves the on-device PURE-top-p (no top-k) sampling fast-path
// — device TopK(256) + full-vocab softmax stats + nlp.SampleTopPFromCandidates, with a host fallback on
// nucleus overflow — reproduces the full-vocab CPU sampler (GOAI_CUDA_TOPK_SAMPLE=0) token-for-token for
// a penalty-free top-p sampler with the same seed. The device path uses a tree-reduced double Zexp vs the
// CPU's sequential f64 sum, so agreement is expected everywhere except astronomically rare ulp-boundary
// coincidences; this small run must match exactly.
func TestGraphLlamaTopPSampleMatchesFallback(t *testing.T) {
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
	// PURE top-p (no top-k) — this is the config that previously fell to the whole-vocab host path.
	mkSampler := func() nlp.TokenSampler {
		return nlp.NewSampler(1234, nlp.WithTemperature(0.8), nlp.WithTopP(0.92))
	}
	gdF, err := NewLlamaQ4KGraphCUDA(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	fast, err := gdF.Generate(prompt, maxNew, mkSampler())
	gdF.Release()
	if err != nil {
		t.Fatalf("fast Generate: %v", err)
	}
	t.Setenv("GOAI_CUDA_TOPK_SAMPLE", "0")
	gdS, err := NewLlamaQ4KGraphCUDA(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	slow, err := gdS.Generate(prompt, maxNew, mkSampler())
	gdS.Release()
	if err != nil {
		t.Fatalf("fallback Generate: %v", err)
	}
	if len(fast) != len(slow) {
		t.Fatalf("length mismatch: fast %d, fallback %d", len(fast), len(slow))
	}
	for i := range fast {
		if fast[i] != slow[i] {
			t.Fatalf("token %d: fast=%d fallback=%d — on-device pure-top-p sampling diverges from full-vocab", i, fast[i], slow[i])
		}
	}
	t.Logf("on-device pure-top-p sampling == full-vocab fallback over %d tokens (TopP=0.92, temp=0.8)", len(fast))
}
