//go:build vulkan

package llamagpu_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestVulkanCooperativeEndToEnd measures what the cooperative M=1 quant shaders are
// worth on a whole model decode rather than an isolated leaf.
//
// Every other measurement of those shaders drives one kernel per recorder submit,
// which is how you isolate a kernel but is not how the decoder runs: llamagpu records
// a whole decode step into one recorder. This runs the real generate loop with the
// cooperative shaders forced off and on, alternating, so the result includes the
// LayerNorm, RoPE, attention and residual work the shaders do not touch — i.e. the
// share of a token they actually move.
func TestVulkanCooperativeEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("124M-parameter model; skipped in -short")
	}
	if !vulkan.Available() {
		t.Skip("vulkan unavailable")
	}
	// TinyLlama-shaped projections (Dim 2048, Hidden 5632) with fewer layers, so the
	// per-layer matmul SHAPES match a deployed model while total size stays testable.
	// The 124M config this started from (Dim 768, Hidden 2048) showed 1.01-1.04x
	// end-to-end because its matmuls are too small for the kernel to matter.
	cfg := nlp.LlamaConfig{
		Vocab: 32000, Ctx: 1024, Dim: 2048, Heads: 16, KVHeads: 4, Layers: 6,
		Hidden: 5632, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*131 + 5) % cfg.Vocab
	}
	const genN = 32

	for _, q := range []struct {
		name string
		qt   gguf.QuantType
		set  func(bool) bool
	}{
		{"Q4_K", gguf.Q4_K, vulkan.SetQ4KCooperative},
		{"Q8_0", gguf.Q8_0, vulkan.SetQ8_0Cooperative},
	} {
		qm, err := nlp.QuantizeLlama(m, q.qt)
		if err != nil {
			t.Fatal(err)
		}
		dec, err := llamagpu.NewQuantVulkan(qm)
		if err != nil {
			qm.Close()
			t.Skipf("%s decoder unavailable: %v", q.name, err)
		}
		sample := func(on bool) float64 {
			prev := q.set(on)
			defer q.set(prev)
			if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil { // warm
				t.Fatal(err)
			}
			start := time.Now()
			if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil {
				t.Fatal(err)
			}
			return float64(genN) / time.Since(start).Seconds()
		}
		var off, on []float64
		for range 3 { // alternate so host drift hits both sides equally
			off = append(off, sample(false))
			on = append(on, sample(true))
		}
		dec.Release()
		qm.Close()
		medOff, medOn := coopMedian(off), coopMedian(on)
		t.Logf("%s end-to-end: scalar %.2f tok/s -> cooperative %.2f tok/s = %.3fx  (off %v on %v)",
			q.name, medOff, medOn, medOn/medOff, off, on)
	}
}

func coopMedian(v []float64) float64 {
	s := append([]float64(nil), v...)
	for i := range s {
		for j := i + 1; j < len(s); j++ {
			if s[j] < s[i] {
				s[i], s[j] = s[j], s[i]
			}
		}
	}
	return s[len(s)/2]
}
