# ADR-0008 — GPU strategy: offload only measured winners

- Status: accepted; amended 2026-07-19 by §T29 after §T338–§T348 measurements
- Date: 2026-07-06
- Relates: §C2/§C3 (cgo gate), §T28–T30, §T42/§T43, §R36–R40

## Context

The user's top priority is GPU/accelerator support for training AND inference.
The naive plan (§T28/§T29) was to offload every op — elementwise, addbias,
softmax, and norms — to the GPU. Early measurement + the cgo gate exposed why
that was wrong for binary/addbias. Later cooperative softmax/norm kernels were
measured at large shapes and overturned the original blanket rejection for that
family (§T338–§T348).

## Decision

1. **Offload only measured winners.** Arithmetic intensity is a screening
   heuristic, not the decision. A route needs representative same-shape
   §C3/§V22 A/B evidence. Binary elementwise/addbias remain on CPU: per-op GPU dispatch and
   transfer lose there (§T28, §T534/§T535). Large softmax/norm kernels remain on
   Metal/Vulkan: at 2048², CPU Softmax/RMSNorm/LayerNorm measured about
   7.76/4.27/6.79 ms versus Metal 1.73/1.68/1.77 ms and Vulkan
   1.76/1.73/1.79 ms (§T29, §T338–§T348). The explicit F32 Metal/Vulkan
   backends currently dispatch every non-empty valid shape; no shape-specific
   CPU threshold is claimed until that threshold is separately A/B-measured.
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

- Roofline predicts likely winners: elementwise is ~1 FLOP/byte while GEMM is
  O(n³)/O(n²). It does not replace measurement; cooperative row reductions can
  still win enough at large shapes to amortize synchronous per-op overhead.
- Reusing the existing metal GEMM + existing CPU autograd made GPU training a
  ~15-line change (configurable tape backend) instead of a kernel rewrite —
  correctness-first, validated against the CPU truth.

## Consequences

- §T28 remains dropped as written. §T29 is superseded: per-op softmax/norm is
  justified by its measured representative shapes; this does not invent an
  unimplemented small-shape router or grant future kernels blanket coverage.
- Full GPU-resident training (no per-op copy) tracked under MPSGraph/CUDA graph
  work (§T42 and a future MPSGraph task).

## Revisit if

A route crosses the §C3/§V22 threshold at a representative shape, or a fused GPU
graph API changes the transfer/dispatch boundary. Re-measure; do not infer the
answer from the op family alone.
