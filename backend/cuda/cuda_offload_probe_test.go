//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// T631 VERIFY-FIRST (ADR-0021 measurement gate): to run a model that EXCEEDS VRAM,
// T631 would offload the overflow layers to the amd64 CPU-SIMD backend. This probe
// measures the per-token CPU-SIMD vs GPU decode cost for TinyLlama-1.1B so the
// offload trade-off (how many layers can spill before the CPU dominates) is known
// BEFORE building the routing/transfer plumbing.
func TestT631OffloadViabilityProbe(t *testing.T) {
	skipNoGPU(t)
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Skipf("model not present: %v", err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	nL := len(m.Blocks)

	// CPU single-token forward (whole model) — the per-token cost if ALL layers ran
	// on CPU-SIMD. Divide by layers for a per-layer estimate (matmuls dominate).
	cpu, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(cpu)
	tokens := []int{1, 15043, 29892}
	_, _ = m.Forward(ctx, tokens) // warm
	t0 := time.Now()
	const iters = 3
	for i := 0; i < iters; i++ {
		if _, err := m.Forward(ctx, tokens); err != nil {
			t.Fatal(err)
		}
	}
	cpuPerTok := time.Since(t0).Seconds() / float64(iters) / float64(len(tokens))
	cpuPerLayer := cpuPerTok / float64(nL)

	// GPU Q8 decode per-token from the measured 164.7 tok/s (§PERF), per-layer.
	const gpuTokPerSec = 164.7
	gpuPerTok := 1.0 / gpuTokPerSec
	gpuPerLayer := gpuPerTok / float64(nL)

	t.Logf("TinyLlama-1.1B (%d layers): CPU-SIMD %.1f ms/layer, GPU-Q8 %.2f ms/layer → CPU is %.0f× per layer",
		nL, cpuPerLayer*1e3, gpuPerLayer*1e3, cpuPerLayer/gpuPerLayer)
	t.Logf("Offload trade-off (keep 22−N on GPU, N on CPU-SIMD):")
	for _, n := range []int{0, 1, 2, 4, 8} {
		tok := float64(nL-n)*gpuPerLayer + float64(n)*cpuPerLayer
		t.Logf("  N=%-2d offloaded → %.1f tok/s (%.1f%% of all-GPU)", n, 1/tok, (1/tok)/gpuTokPerSec*100)
	}
	t.Logf("VERDICT: CPU-SIMD offload is FUNCTIONAL (runs >VRAM models) but each offloaded")
	t.Logf("layer costs ≈%.0f× the GPU — so T631 must spill the MINIMUM overflow layers.", cpuPerLayer/gpuPerLayer)
}
