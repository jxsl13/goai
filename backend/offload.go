package backend

import (
	"fmt"
	"sort"
)

// OffloadPlan is the layer→backend assignment policy behind C24 low-VRAM
// execution (§T631): it decides which of a model's layers run on the fast
// device(s) and which spill to a host/CPU fallback when the model does not fit
// in device memory. It is the pure scheduling half of low-VRAM offload — it
// consumes measured sizes and budgets and returns an assignment; the caller
// applies the assignment through Context.WithLayerBackend (§T630) and supplies
// the per-backend memory budgets from a device probe.
//
// # For the AI professional
//
// Given each layer's device-memory footprint and a per-backend byte budget,
// PlanOffload greedily keeps the earliest (or, with Hotness, the hottest)
// layers resident on the preferred backends in order, packing each to its
// budget, then assigns every remaining layer to the Fallback backend (treated
// as unbounded — host RAM). This mirrors llama.cpp's n-gpu-layers behavior:
// as many layers on the accelerator as fit, the rest on the CPU, always
// functional (§C24) rather than an OOM failure. The plan is OFF the hot path
// (§C23/ADR-0021: a split on a device that fits the whole model is a
// transfer≫compute loss) — it exists to make oversized models RUN, not to
// speed up models that already fit.
//
// # For everyone else
//
// A big model may not fit in your graphics card's memory. This function is the
// planner that puts as many of the model's layers on the fast card as fit and
// the leftover layers on the regular CPU/RAM, so the model still runs (just
// slower) instead of failing.
//
// Further reading: the llama.cpp "-ngl / --n-gpu-layers" partial-offload design
// (github.com/ggerganov/llama.cpp) is the canonical precedent; ADR-0021 records
// why the split is functionality, not speed, on fits-in-VRAM devices.
type OffloadPlan struct {
	// Layers is the resolved per-layer backend assignment, one Name per input
	// layer in the original layer order.
	Layers []Name
	// OnFallback counts how many layers spilled to the Fallback backend.
	OnFallback int
}

// MemoryProber is an OPTIONAL capability a Backend may implement to report the
// device memory currently available to it, so PlanOffload's per-backend Budgets
// can be measured rather than guessed (§T631). It is not part of the core
// Backend interface: a backend that cannot query its device (the host/CPU and
// reference backends, or a GPU backend on a platform without a memory-info API)
// simply does not implement it, and ProbeBudgets omits it. CUDA reports live
// free VRAM, Metal reports its remaining recommended working set, and Vulkan
// reports VK_EXT_memory_budget when the driver exposes it.
type MemoryProber interface {
	// AvailableMemory reports the free device memory in bytes. ok is false when
	// the amount is unknown/unqueryable, in which case the caller treats the
	// backend as having no measured device budget (so its layers spill to the
	// fallback rather than risk an out-of-memory placement).
	AvailableMemory() (bytes int64, ok bool)
}

// ProbeBudgets builds the per-backend Budgets map for PlanOffload by querying
// each named registered backend through the optional MemoryProber capability.
// A name that is not registered, does not implement MemoryProber, or reports
// ok=false (or a non-positive amount) is omitted — treated as having no device
// budget, so PlanOffload spills its share to the fallback (§C24: functional,
// never an out-of-memory failure). The returned map is safe to hand straight to
// OffloadConfig.Budgets.
func ProbeBudgets(names ...Name) map[Name]int64 {
	budgets := make(map[Name]int64)
	for _, name := range names {
		b, ok := Get(name)
		if !ok {
			continue
		}
		mp, isProber := b.(MemoryProber)
		if !isProber {
			continue
		}
		if bytes, known := mp.AvailableMemory(); known && bytes > 0 {
			budgets[name] = bytes
		}
	}
	return budgets
}

// OffloadConfig parameterises PlanOffload.
type OffloadConfig struct {
	// LayerBytes is each layer's device-memory footprint (weights + activation
	// scratch), in bytes, in model layer order. Required.
	LayerBytes []int64
	// Budgets maps a backend to the byte budget available on it (from a device
	// memory probe). Layers are placed on the Prefer backends up to these
	// budgets. A backend absent from Budgets (or with a non-positive budget) is
	// treated as having no room.
	Budgets map[Name]int64
	// Prefer is the order in which fast backends are filled (e.g. the GPU
	// first). Layers pack onto Prefer[0] until its budget is exhausted, then
	// Prefer[1], and so on.
	Prefer []Name
	// Fallback is the always-available backend that absorbs every layer that did
	// not fit on a Prefer backend — the host/CPU path (§C24: never a hard
	// failure). Treated as unbounded. Required.
	Fallback Name
	// Hotness, if non-nil, ranks layers by importance (higher = keep on the fast
	// backend first); it must have one entry per layer. When nil, layers are
	// kept in natural order (layer 0 first), which matches the common
	// "first-N-layers-on-GPU" policy.
	Hotness []float64
}

// PlanOffload resolves an OffloadConfig into a per-layer backend assignment.
//
// It never fails to place a layer: any layer that does not fit a Prefer backend
// goes to Fallback, so an oversized model still gets a runnable plan (§C24). It
// returns an error only for malformed input (empty LayerBytes, a Hotness of the
// wrong length, or an empty Fallback).
//
// The returned Layers slice is in the original layer order regardless of
// Hotness; Hotness only affects WHICH layers win the scarce fast-backend
// budget, not the output ordering.
func PlanOffload(cfg OffloadConfig) (OffloadPlan, error) {
	n := len(cfg.LayerBytes)
	if n == 0 {
		return OffloadPlan{}, fmt.Errorf("backend: PlanOffload needs at least one layer")
	}
	if cfg.Fallback == "" {
		return OffloadPlan{}, fmt.Errorf("backend: PlanOffload needs a Fallback backend (§C24)")
	}
	if cfg.Hotness != nil && len(cfg.Hotness) != n {
		return OffloadPlan{}, fmt.Errorf("backend: PlanOffload Hotness has %d entries, want %d", len(cfg.Hotness), n)
	}

	// Visit layers in descending hotness (stable, so equal-hotness layers keep
	// natural order); with no hotness this is just 0,1,2,… .
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	if cfg.Hotness != nil {
		//perfscan:ignore PS3002,PS6009 one-time layer-plan sort, tiny n (layers) | one-time layer-plan sort, tiny n
		sort.SliceStable(order, func(a, b int) bool {
			return cfg.Hotness[order[a]] > cfg.Hotness[order[b]]
		})
	}

	// Remaining budget per preferred backend (copy so we don't mutate the input).
	remaining := make(map[Name]int64, len(cfg.Prefer))
	for _, name := range cfg.Prefer {
		if b := cfg.Budgets[name]; b > 0 {
			remaining[name] = b
		}
	}

	plan := OffloadPlan{Layers: make([]Name, n)}
	for i := range plan.Layers {
		plan.Layers[i] = cfg.Fallback // default; overwritten if it fits a Prefer backend
	}
	plan.OnFallback = n

	for _, li := range order {
		size := cfg.LayerBytes[li]
		placed := false
		for _, name := range cfg.Prefer {
			if remaining[name] >= size { // fits on this fast backend
				remaining[name] -= size
				plan.Layers[li] = name
				plan.OnFallback--
				placed = true
				break
			}
		}
		_ = placed // a layer that fits nowhere stays on Fallback (already set)
	}
	return plan, nil
}
