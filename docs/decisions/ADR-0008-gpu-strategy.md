# ADR-0008 — GPU strategy: offload compute-bound ops, not memory-bound ones

- Status: accepted (autonomous loop, §T30)
- Date: 2026-07-06
- Relates: §C2/§C3 (cgo gate), §T28–T30, §T42/§T43, §R36–R40

## Context

The user's top priority is GPU/accelerator support for training AND inference.
The naive plan (§T28/§T29) was to offload every op — elementwise, addbias,
softmax — to the GPU. Measurement + the cgo gate expose why that is wrong.

## Decision

1. **Offload only compute-bound ops** (GEMM/conv), where FLOPs dominate transfer.
   Memory-bound ops (elementwise, addbias, bias, norms) stay on CPU: per-op GPU
   offload pays a host↔device copy of the whole tensor for O(n) work → it loses
   to the CPU and **fails the §C3 gate**. §T28/§T29 (per-op offload) are dropped.
2. **GPU training = dispatch the training GEMMs to the GPU backend.** A tape
   bound to the metal backend (`NewTapeOn(metal)`) runs the forward GEMM and both
   backward GEMMs of the matmul VJP on the GPU; all other ops fall back to CPU
   (§I4). This delivered GPU-accelerated MLP training, converging identically to
   the CPU reference (T30), with zero new kernels.
3. **The durable high-throughput path is graph residency** — keep intermediates
   on the device across a whole forward/backward graph (Apple MPSGraph exposes
   GPU autodiff, §R36; CUDA graphs; Vulkan command buffers). That is the vehicle
   for large-model GPU training/inference (§T42 CUDA, MPSGraph expansion), where
   the per-op transfer tax vanishes.

## Rationale

- Roofline: elementwise is ~1 FLOP/byte → PCIe/UMA copy >> compute; GEMM is
  O(n³)/O(n²) → compute >> copy. The gate correctly rejects the former.
- Reusing the existing metal GEMM + existing CPU autograd made GPU training a
  ~15-line change (configurable tape backend) instead of a kernel rewrite —
  correctness-first, validated against the CPU truth.

## Consequences

- §T28/§T29 removed as written; GPU op coverage grows only where compute-bound
  or where graph residency amortizes transfer.
- Full GPU-resident training (no per-op copy) tracked under MPSGraph/CUDA graph
  work (§T42 and a future MPSGraph task).

## Revisit if

A fused GPU graph API (MPSGraph/CUDA graphs) lands — then whole-subgraph
residency replaces per-op dispatch and memory-bound ops ride along for free.
