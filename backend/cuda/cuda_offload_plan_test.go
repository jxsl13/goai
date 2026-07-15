//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// The T631 layer-assignment policy must split a model's layers correctly across all the
// budget regimes: whole model fits, partial spill, everything spills, and the guard when
// the fixed (non-layer) state alone overflows the budget.
func TestCUDAPlanOffload(t *testing.T) {
	const perLayer = 60 << 20 // 60 MiB/layer
	const fixed = 500 << 20   // 500 MiB embeddings+head
	const nL = 22

	cases := []struct {
		name     string
		budget   uint64
		wantGPU  int
		wantCPU  int
		wantFits bool
	}{
		{"fits-whole", fixed + nL*perLayer, 22, 0, true},
		{"fits-with-slack", fixed + 100*perLayer, 22, 0, true},
		{"partial-spill", fixed + 10*perLayer, 10, 12, false},
		{"tight-one-layer", fixed + 1*perLayer, 1, 21, false},
		{"only-fixed-fits", fixed + perLayer/2, 0, 22, false},
		{"fixed-overflows", fixed / 2, 0, 22, false},
	}
	for _, c := range cases {
		p := cuda.PlanOffload(nL, perLayer, fixed, c.budget)
		if p.GPULayers != c.wantGPU || p.CPULayers != c.wantCPU || p.Fits != c.wantFits {
			t.Errorf("%s: budget=%dMiB → %+v, want GPU=%d CPU=%d Fits=%v",
				c.name, c.budget>>20, p, c.wantGPU, c.wantCPU, c.wantFits)
		}
	}
	// Degenerate inputs are treated as "fits" (nothing to place).
	if p := cuda.PlanOffload(0, perLayer, fixed, 0); !p.Fits {
		t.Errorf("nLayers=0 should report Fits, got %+v", p)
	}
	if p := cuda.PlanOffload(nL, 0, fixed, 0); !p.Fits {
		t.Errorf("perLayerBytes=0 should report Fits, got %+v", p)
	}
	t.Log("PlanOffload splits correctly across fits / partial / all-CPU / underflow regimes")
}

// The Q8 per-layer VRAM estimate for TinyLlama-1.1B geometry must be in the ballpark of
// the measured footprint (≈62.5 MiB/layer total; the estimate is weights-only so it is a
// bit lower), and the estimate must drive a sensible plan under a simulated VRAM cap.
func TestCUDAEstimateLayerVRAMQ8(t *testing.T) {
	// TinyLlama-1.1B: dim 2048, heads 32 → headDim 64, kvHeads 4 → kvDim 256, hidden 5632.
	const dim, kvDim, hidden, nL = 2048, 256, 5632, 22
	perLayer := cuda.EstimateLayerVRAMQ8(dim, kvDim, hidden)
	mib := perLayer >> 20
	// Weights-only Q8 for these dims is ≈47 MiB; assert a sane 30–63 MiB band (below the
	// 62.5 MiB measured total that also includes KV + scratch).
	if mib < 30 || mib > 63 {
		t.Fatalf("EstimateLayerVRAMQ8 = %d MiB, out of the expected 30–63 MiB band", mib)
	}
	t.Logf("TinyLlama-1.1B Q8 weights ≈ %d MiB/layer (measured total ≈62.5 incl KV+scratch)", mib)

	// Simulate a 1 GiB VRAM cap with 256 MiB fixed state → only some layers fit.
	const fixed = 256 << 20
	plan := cuda.PlanOffload(nL, perLayer, fixed, 1<<30)
	if plan.Fits || plan.GPULayers < 1 || plan.GPULayers >= nL {
		t.Fatalf("1 GiB cap should force a partial spill of TinyLlama, got %+v", plan)
	}
	if plan.GPULayers+plan.CPULayers != nL {
		t.Fatalf("plan layers must sum to %d, got %+v", nL, plan)
	}
	t.Logf("1 GiB cap → keep %d layers on GPU, spill %d to CPU-SIMD (T631)", plan.GPULayers, plan.CPULayers)

	// On the real device, TinyLlama-1.1B (≈62.5 MiB×22 ≈ 1.4 GiB) fits with headroom.
	got, err := cuda.PlanOffloadForDevice(nL, perLayer, fixed, 512<<20)
	if err != nil {
		t.Skipf("no device VRAM: %v", err)
	}
	if !got.Fits {
		t.Logf("device plan for TinyLlama: %+v (did not fit — small/busy GPU?)", got)
	} else {
		t.Logf("device plan for TinyLlama: fits fully on GPU (%d layers, 0 spilled)", got.GPULayers)
	}
}
