# ADR-0018 — Zero-copy UMA for GPU ops (page-aligned storage + bytesNoCopy)

Status: **REJECTED after measurement** (2026-07-12, §T348). Built end-to-end, then
reverted: a clean same-session A/B showed **no speedup** — the host↔device copy is
not the bottleneck the earlier notes assumed. Kept as a record so it is not
re-attempted without new evidence. See "Outcome" below.

## Outcome (the reason it was rejected)

Phase 1 was implemented in full and verified correct (page-aligned heap allocator +
Metal `bytesNoCopy` auto-detection, applied to RMSNorm; cross-reference green,
zero-copy path confirmed active via instrumentation — pointer and length both
page-aligned, `ok=1`). Then a controlled A/B at 2048×2048 (the size the whole
norm/softmax family was benchmarked at), same session, back-to-back:

| RMSNorm/metal 2048² | median ns/op |
|---|---|
| copy path (bytesNoCopy forced off) | ≈1.75 ms |
| zero-copy (bytesNoCopy active)     | ≈1.73 ms |

Within noise — **eliminating both 16 MB memcpys changed nothing.** So the ≈1.7 ms
floor these ops hit after §T345–§T347 is NOT the host↔device copy; it is per-op GPU
dispatch/`waitUntilCompleted` latency and/or reduction-barrier serialization (the
cooperative kernels move ≈48 MB at only ≈25 GB/s, far under the ≈200 GB/s the
hardware can do, so they are latency/occupancy-bound, not copy- or bandwidth-bound).

Consequence: the "remaining floor is the memcpy" claim recorded in the §T345/§T346/
§T347 notes and `docs/benchmarking.md` was an **unverified assumption** and is
wrong (corrected there and logged in §B). The real next lever for this family is
**reducing per-op GPU round-trips** — batching several ops into one command buffer
with barriers (as §T343 did for conv), or a persistent-encoder / graph submission —
not zero-copy and not more per-kernel work. Zero-copy would only matter once the
op is copy-bound, which it is not here.

Because it delivered no measured win while adding a global allocator change (unsafe,
per-large-tensor page waste), it was reverted per §C3 / V-CGO ("do not merge a
non-winning optimization"). Re-open only with evidence that an op is copy-bound.

## Context

On Apple Silicon (Metal) and MoltenVK (Vulkan on the same hardware) a
`StorageModeShared` / `HOST_VISIBLE` GPU buffer **is** host RAM — the GPU and CPU
address the same physical pages (Unified Memory Architecture). Yet every GPU op
did two `memcpy`s: host tensor slice → buffer `.contents` (upload) before the
dispatch, and buffer `.contents` → host tensor slice (download) after.

§T335–§T347 drove the compute kernels of the whole norm/softmax family down to
their bandwidth limit. At that point the benchmarks showed the op time is
dominated by those two memcpys, not the kernel: a 2048×2048 RMSNorm/softmax/
LayerNorm sits at a ≈1.7 ms floor that is exactly `32 MB / (single-thread memcpy
BW ≈19 GB/s)`. Copying host memory into host memory to run a kernel that could
read the original is pure overhead on UMA.

## Decision

Skip the copies by letting the GPU buffer **wrap the tensor's own storage**:
`newBufferWithBytesNoCopy` (Metal) over the tensor's `[]float32` backing memory.
The kernel then reads inputs and writes outputs in place.

`newBufferWithBytesNoCopy` requires the host pointer to be page-aligned (16 KiB on
Apple Silicon) and, in practice, the length to be a page multiple. Two parts:

1. **Page-aligned tensor storage (pure Go, CGO0-safe).** The default heap
   allocator page-aligns allocations at or above a threshold (`allocAlignThreshold`)
   by over-allocating and returning an aligned sub-slice. Small tensors (biases,
   scales) stay unaligned — a page of waste each is not worth it and their copies
   are negligible. This is behaviour-preserving: still a zeroed `[]float32`, only
   the backing array is larger and offset. No cgo — the pure-Go build is untouched.

2. **Auto-detected zero-copy in the Metal bridge.** A helper wraps a host pointer
   with `bytesNoCopy` **iff** it is page-aligned and its length is a page multiple;
   otherwise it falls back to the existing pooled-buffer + memcpy path (§T336). No
   op signatures change: the C side inspects the pointer it already receives. On
   the zero-copy path the upload and/or download memcpy is skipped.

Correctness/safety:
- `[]float32` holds no Go pointers, so passing `&slice[0]` to C obeys the cgo
  pointer rules; the buffer is released within the synchronous op (the tensor slice
  is live on the Go stack throughout `waitUntilCompleted`), so the memory stays
  valid and the GC neither moves (Go heap objects don't move) nor frees it.
- `deallocator:nil` — Metal must not free Go-owned memory.
- Kernels still write every output element they expose, so wrapping in place is
  as safe as the pooled path (no reliance on zero-fill; §T336 invariant holds).

## Phasing

- **Phase 1 (§T348, this ADR):** the aligned allocator + Metal zero-copy applied to
  the memory-bound norm/softmax family (the ops proven at the memcpy floor).
- **Phase 2:** extend the Metal zero-copy helper to the remaining ops (matmul,
  attention, conv, elementwise) — same helper, mechanical.
- **Phase 3:** Vulkan zero-copy via `VK_EXT_external_memory_host` (import the host
  pointer as device memory) where MoltenVK/the driver supports it; else Vulkan
  keeps the pooled path. Gated on capability, reference-fallback-safe.

## Alternatives rejected

- **Device-resident tensor type (PyTorch-style):** biggest win but a large
  cross-cutting L0+autograd+backend change; deferred, this ADR is the incremental
  path to it.
- **Aligning every tensor:** wastes a page on each of the many tiny tensors;
  thresholded alignment avoids it.
- **C/`posix_memalign`-backed storage:** would break the `CGO_ENABLED=0` build
  (§V7) — the aligned allocator must be pure Go.
