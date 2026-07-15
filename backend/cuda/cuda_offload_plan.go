//go:build cuda && cgo && (linux || windows)

package cuda

import "fmt"

// Layer-assignment policy for T631 (run a model that EXCEEDS device VRAM by spilling
// overflow layers to the CPU-SIMD path). The offload MECHANISM — a decode split with
// dual GPU/CPU KV caches, hidden state crossing the boundary — is already proven
// (TestCUDAHybridDecode); this is the POLICY that decides HOW MANY layers to spill,
// driven by the measured VRAM budget (MemInfo). It keeps the FIRST gpuLayers resident
// on the GPU and offloads the rest to the CPU, which minimises the number of spilled
// layers (each offloaded layer costs ≈30× a GPU layer, §PERF-T631 — so spill the
// fewest possible). Wiring this into the real serving API is the T630 shared-executor
// step (coordinate); the policy + estimator here are backend-local and dependency-free.

const q8ScaleOverheadNum, q8ScaleOverheadDen = 36, 32 // 1 int8 byte/weight + 4-byte f32 scale per 32-block = ×36/32

// EstimateLayerVRAMQ8 returns the resident VRAM (bytes) one Llama/Qwen transformer
// layer's WEIGHTS occupy as resident Q8 (the 7 projections q,k,v,o,gate,up,down), from
// the model geometry: dim (model width), kvDim (kvHeads·headDim), hidden (FFN width).
// Q8 stores 1 int8 byte per weight plus one f32 scale per 32-block, i.e. ×36/32 bytes.
// This is weights-only; the per-layer KV cache and activation scratch are separate (the
// authoritative total is the measured footprint — ≈62.5 MiB/layer for TinyLlama-1.1B Q8,
// §PERF-T631 — so callers should prefer a measured perLayerBytes when they have one).
func EstimateLayerVRAMQ8(dim, kvDim, hidden int) uint64 {
	if dim <= 0 || kvDim < 0 || hidden <= 0 {
		return 0
	}
	// weight element counts: q(dim·dim) k(dim·kvDim) v(dim·kvDim) o(dim·dim)
	// gate(dim·hidden) up(dim·hidden) down(hidden·dim).
	elems := uint64(dim)*uint64(dim)*2 + uint64(dim)*uint64(kvDim)*2 + uint64(dim)*uint64(hidden)*3
	return elems * q8ScaleOverheadNum / q8ScaleOverheadDen
}

// OffloadPlan is the result of PlanOffload: how a model's layers are split between the
// GPU (the first GPULayers) and the CPU-SIMD overflow path (the remaining CPULayers).
type OffloadPlan struct {
	GPULayers int  // layers kept resident on the GPU (the hottest — the first N)
	CPULayers int  // overflow layers spilled to the CPU-SIMD path
	Fits      bool // true iff the whole model fits on the GPU (CPULayers == 0)
}

// PlanOffload decides the GPU/CPU layer split for a model of nLayers layers, each
// costing perLayerBytes of resident VRAM, plus fixedBytes of non-layer resident state
// (embedding table, output head, and whatever KV/scratch headroom the caller reserves),
// within a budgetBytes VRAM budget. It fills the budget with as many layers as fit —
// keeping the first (hottest) on the GPU — and reports the rest as CPU overflow, so the
// MINIMUM number of layers spill (§PERF-T631: each offloaded layer is ≈30× a GPU layer).
// If even the fixed state overflows the budget, GPULayers is 0 (all layers on CPU).
func PlanOffload(nLayers int, perLayerBytes, fixedBytes, budgetBytes uint64) OffloadPlan {
	if nLayers <= 0 || perLayerBytes == 0 {
		return OffloadPlan{Fits: true}
	}
	var avail uint64
	if budgetBytes > fixedBytes {
		avail = budgetBytes - fixedBytes
	}
	gpu := int(avail / perLayerBytes)
	if gpu > nLayers {
		gpu = nLayers
	}
	return OffloadPlan{GPULayers: gpu, CPULayers: nLayers - gpu, Fits: gpu == nLayers}
}

// PlanOffloadForDevice is PlanOffload against the device's live free VRAM (MemInfo),
// reserving reserveBytes of that free space as headroom for the KV cache, activation
// scratch and driver overhead the raw layer/fixed estimates omit. Errors if no CUDA
// device VRAM is reported.
func PlanOffloadForDevice(nLayers int, perLayerBytes, fixedBytes, reserveBytes uint64) (OffloadPlan, error) {
	free, _ := MemInfo()
	if free == 0 {
		return OffloadPlan{}, fmt.Errorf("cuda: PlanOffloadForDevice: no device VRAM reported")
	}
	var budget uint64
	if free > reserveBytes {
		budget = free - reserveBytes
	}
	return PlanOffload(nLayers, perLayerBytes, fixedBytes, budget), nil
}
