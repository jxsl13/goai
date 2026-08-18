//go:build darwin && cgo

package llamagpu_test

import (
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestMetalDecoderTopKSampleMatchesFallback proves that Metal's resident TopKN path preserves the
// complete public Generate contract, including the sampler's RNG consumption and token sequence.
func TestMetalDecoderTopKSampleMatchesFallback(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 2048, Ctx: 96, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	model, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 9, 42, 17}
	newSampler := func() nlp.TokenSampler {
		return nlp.NewSampler(1234,
			nlp.WithTemperature(0.8), nlp.WithTopK(40), nlp.WithTopP(0.92),
		)
	}
	const maxNew = 48

	fastDecoder, err := llamagpu.New(model)
	if err != nil {
		t.Fatal(err)
	}
	fast, err := fastDecoder.Generate(prompt, maxNew, newSampler())
	fastDecoder.Release()
	if err != nil {
		t.Fatal(err)
	}

	t.Setenv("GOAI_DEVICE_TOPK_SAMPLE", "0")
	fallbackDecoder, err := llamagpu.New(model)
	if err != nil {
		t.Fatal(err)
	}
	fallback, err := fallbackDecoder.Generate(prompt, maxNew, newSampler())
	fallbackDecoder.Release()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(fast, fallback) {
		for i := range min(len(fast), len(fallback)) {
			if fast[i] != fallback[i] {
				t.Fatalf("token %d: resident=%d fallback=%d", i, fast[i], fallback[i])
			}
		}
		t.Fatalf("generated lengths differ: resident=%d fallback=%d", len(fast), len(fallback))
	}
}
