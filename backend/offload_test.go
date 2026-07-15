package backend

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// proberBackend is a minimal Backend that also implements MemoryProber, for
// testing ProbeBudgets.
type proberBackend struct {
	name  Name
	avail int64
	ok    bool
}

func (p *proberBackend) Name() Name                             { return p.name }
func (p *proberBackend) Device() tensor.Device                  { return tensor.CPU() }
func (p *proberBackend) Synchronize() error                     { return nil }
func (p *proberBackend) Kernel(Op, tensor.Dtype) (Kernel, bool) { return nil, false }
func (p *proberBackend) AvailableMemory() (int64, bool)         { return p.avail, p.ok }

// TestProbeBudgets: only registered backends that implement MemoryProber and
// report a known, positive amount contribute a budget; the rest are omitted.
func TestProbeBudgets(t *testing.T) {
	known := &proberBackend{name: "probe-known", avail: 8_000_000_000, ok: true}
	unknown := &proberBackend{name: "probe-unknown", avail: 0, ok: false} // can't query
	nonProber := &spyBackend{name: "probe-nonprober", kern: scaleKernel}  // no MemoryProber
	Register(known)
	Register(unknown)
	Register(nonProber)

	got := ProbeBudgets(known.Name(), unknown.Name(), nonProber.Name(), "probe-unregistered")
	if len(got) != 1 {
		t.Fatalf("ProbeBudgets returned %d budgets, want 1 (only the known prober): %v", len(got), got)
	}
	if got[known.Name()] != 8_000_000_000 {
		t.Fatalf("known prober budget = %d, want 8e9", got[known.Name()])
	}
	if _, in := got[unknown.Name()]; in {
		t.Fatal("unknown-amount prober should be omitted")
	}
	if _, in := got[nonProber.Name()]; in {
		t.Fatal("non-prober backend should be omitted")
	}
}

// TestProbeBudgetsFeedsPlan: the probe→plan pipeline — a probed budget drives
// PlanOffload, so an 8-layer model that half-fits keeps 4 layers on the device.
func TestProbeBudgetsFeedsPlan(t *testing.T) {
	dev := &proberBackend{name: "probe-plan-dev", avail: 40, ok: true} // holds 4 layers of 10
	Register(dev)
	budgets := ProbeBudgets(dev.Name())
	plan, err := PlanOffload(OffloadConfig{
		LayerBytes: []int64{10, 10, 10, 10, 10, 10, 10, 10},
		Budgets:    budgets,
		Prefer:     []Name{dev.Name()},
		Fallback:   CPU,
	})
	if err != nil {
		t.Fatal(err)
	}
	if countOn(plan.Layers, dev.Name()) != 4 || plan.OnFallback != 4 {
		t.Fatalf("probe→plan: %d on device / %d spilled, want 4/4", countOn(plan.Layers, dev.Name()), plan.OnFallback)
	}
}

// ExamplePlanOffload plans a 4-layer model onto a GPU whose memory budget holds
// only two layers: the first two stay on the fast backend and the rest spill to
// the CPU fallback, so the oversized model still runs (§C24).
func ExamplePlanOffload() {
	plan, _ := PlanOffload(OffloadConfig{
		LayerBytes: []int64{10, 10, 10, 10},
		Budgets:    map[Name]int64{Metal: 25}, // room for 2 layers of 10
		Prefer:     []Name{Metal},
		Fallback:   CPU,
	})
	fmt.Println("layer0 on GPU:", plan.Layers[0] == Metal, "| spilled to CPU:", plan.OnFallback)
	// Output: layer0 on GPU: true | spilled to CPU: 2
}

// countOn returns how many assignments in plan equal name.
func countOn(layers []Name, name Name) int {
	n := 0
	for _, l := range layers {
		if l == name {
			n++
		}
	}
	return n
}

// TestPlanOffloadFitsAllOnGPU: when the GPU budget covers every layer, no layer
// spills to the fallback (the fits-in-VRAM case — the whole model stays fast).
func TestPlanOffloadFitsAllOnGPU(t *testing.T) {
	plan, err := PlanOffload(OffloadConfig{
		LayerBytes: []int64{10, 10, 10, 10},
		Budgets:    map[Name]int64{Metal: 100},
		Prefer:     []Name{Metal},
		Fallback:   CPU,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.OnFallback != 0 {
		t.Fatalf("OnFallback = %d, want 0 (all fit on GPU)", plan.OnFallback)
	}
	if got := countOn(plan.Layers, Metal); got != 4 {
		t.Fatalf("layers on Metal = %d, want 4", got)
	}
}

// TestPlanOffloadPartialSpill: a budget that holds only some layers keeps the
// earliest layers on the GPU and spills the rest to the CPU (llama.cpp
// first-N-on-GPU behavior), and every layer is still assigned (§C24 functional).
func TestPlanOffloadPartialSpill(t *testing.T) {
	// Budget 25, layers 10 each → 2 fit on GPU, 2 spill.
	plan, err := PlanOffload(OffloadConfig{
		LayerBytes: []int64{10, 10, 10, 10},
		Budgets:    map[Name]int64{Metal: 25},
		Prefer:     []Name{Metal},
		Fallback:   CPU,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.OnFallback != 2 {
		t.Fatalf("OnFallback = %d, want 2", plan.OnFallback)
	}
	// Natural order (no hotness): the first two layers keep the GPU.
	if plan.Layers[0] != Metal || plan.Layers[1] != Metal {
		t.Fatalf("layers 0,1 = %v,%v, want the fast backend", plan.Layers[0], plan.Layers[1])
	}
	if plan.Layers[2] != CPU || plan.Layers[3] != CPU {
		t.Fatalf("layers 2,3 = %v,%v, want the fallback", plan.Layers[2], plan.Layers[3])
	}
	// Every layer is assigned to a real backend.
	for i, l := range plan.Layers {
		if l == "" {
			t.Fatalf("layer %d unassigned", i)
		}
	}
}

// TestPlanOffloadNoGPU: no fast budget at all → the whole model runs on the
// fallback, functional (§C24), never a failure.
func TestPlanOffloadNoGPU(t *testing.T) {
	plan, err := PlanOffload(OffloadConfig{
		LayerBytes: []int64{10, 10, 10},
		Budgets:    map[Name]int64{}, // no device memory
		Prefer:     []Name{Metal},
		Fallback:   CPU,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.OnFallback != 3 || countOn(plan.Layers, CPU) != 3 {
		t.Fatalf("want all 3 layers on CPU fallback, got OnFallback=%d", plan.OnFallback)
	}
}

// TestPlanOffloadHotnessWinsBudget: with a scarce budget, the HOTTEST layers
// (not the first) win the fast backend, but the output stays in layer order.
func TestPlanOffloadHotnessWinsBudget(t *testing.T) {
	// Budget holds exactly 1 layer of size 10; layer 3 is the hottest.
	plan, err := PlanOffload(OffloadConfig{
		LayerBytes: []int64{10, 10, 10, 10},
		Budgets:    map[Name]int64{Metal: 10},
		Prefer:     []Name{Metal},
		Fallback:   CPU,
		Hotness:    []float64{0.1, 0.2, 0.3, 0.9}, // layer 3 hottest
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.OnFallback != 3 {
		t.Fatalf("OnFallback = %d, want 3", plan.OnFallback)
	}
	if plan.Layers[3] != Metal {
		t.Fatalf("hottest layer 3 = %v, want the fast backend", plan.Layers[3])
	}
	for i := 0; i < 3; i++ {
		if plan.Layers[i] != CPU {
			t.Fatalf("cooler layer %d = %v, want the fallback", i, plan.Layers[i])
		}
	}
}

// TestPlanOffloadMultiPrefer: two fast backends fill in Prefer order before the
// fallback.
func TestPlanOffloadMultiPrefer(t *testing.T) {
	plan, err := PlanOffload(OffloadConfig{
		LayerBytes: []int64{10, 10, 10, 10},
		Budgets:    map[Name]int64{CUDA: 15, Metal: 15},
		Prefer:     []Name{CUDA, Metal},
		Fallback:   CPU,
	})
	if err != nil {
		t.Fatal(err)
	}
	// CUDA holds 1 (10, next 10 won't fit 15-10=5), Metal holds 1, 2 spill.
	if countOn(plan.Layers, CUDA) != 1 || countOn(plan.Layers, Metal) != 1 || plan.OnFallback != 2 {
		t.Fatalf("plan = %v (OnFallback %d), want 1 CUDA + 1 Metal + 2 CPU", plan.Layers, plan.OnFallback)
	}
}

// TestPlanOffloadErrors: malformed inputs are rejected.
func TestPlanOffloadErrors(t *testing.T) {
	if _, err := PlanOffload(OffloadConfig{Fallback: CPU}); err == nil {
		t.Fatal("empty LayerBytes should error")
	}
	if _, err := PlanOffload(OffloadConfig{LayerBytes: []int64{1}}); err == nil {
		t.Fatal("empty Fallback should error")
	}
	if _, err := PlanOffload(OffloadConfig{LayerBytes: []int64{1, 2}, Fallback: CPU, Hotness: []float64{1}}); err == nil {
		t.Fatal("mismatched Hotness length should error")
	}
}
