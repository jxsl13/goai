//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// T631 VRAM-budget probe: measure the actual resident-model GPU footprint so the
// offload policy knows when it must spill layers to CPU-SIMD. Builds the Q8
// TinyLlama-1.1B resident model and reads free VRAM before/after via cudaMemGetInfo,
// then extrapolates the all-GPU model-size ceiling on this 12 GB card.
func TestCUDAVRAMBudgetProbe(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	skipNoGPU(t)
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Skipf("model not present: %v", err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	const mib = 1024 * 1024

	free0, total := cuda.MemInfo()
	q8 := buildResidentLlamaQ8(t, m) // Q8-resident TinyLlama-1.1B (22 layers)
	free1, _ := cuda.MemInfo()
	footprint := int64(free0) - int64(free1)
	q8.free()
	free2, _ := cuda.MemInfo()

	nL := len(m.Blocks)
	t.Logf("Device VRAM: %d MiB total, %d MiB free at start", total/mib, free0/mib)
	t.Logf("Q8 TinyLlama-1.1B (22L) resident footprint: %d MiB (≈%.1f MiB/layer)",
		footprint/mib, float64(footprint)/mib/float64(nL))
	t.Logf("Freed on release: %d MiB (free now %d MiB)", (int64(free2)-int64(free1))/mib, free2/mib)
	// Extrapolate: how big an all-GPU Q8 model fits, and when T631 offload activates.
	perLayer := float64(footprint) / float64(nL)
	if perLayer > 0 {
		fitLayers := float64(free0) * 0.85 / perLayer // 85% of free for weights, rest for KV+activations
		t.Logf("T631 activation: ≈%.0f Q8 layers fit all-GPU (85%% of free VRAM); a model with MORE",
			fitLayers)
		t.Logf("layers/wider dims than that must offload the overflow to CPU-SIMD (T631, ≈30×/layer).")
	}
	if footprint <= 0 {
		t.Skip("footprint non-positive (async alloc / driver caching) — MemInfo still exposed")
	}
}
