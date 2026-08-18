# ADR-0019 — Device-resident tensors + command-buffer batching

Status: accepted-design (2026-07-12, §T366). Design + phasing recorded; implementation
is high-blast-radius (L0 storage + the Execute contract) and proceeds in the phases
below, each its own verified change. No behaviour changes until Phase 2 lands.

## Context (measured)

Two independent findings from the §T350–§T361 real-workload measurements point at the
same root cause — the backends execute **one op per command buffer, synchronously**
(encode → commit → `waitUntilCompleted` → download), so every op pays a full CPU↔GPU
round-trip (≈200 µs on this hardware, §T352):

1. **Decode is dispatch-bound (§T360/§T361).** A single-token decode step is ≈95 tiny
   ops → ≈95 round-trips ≈ 19 ms, so Metal is ≈2.3× *slower* than the CPU for small
   models (the crossover is at ≈dim 1024). Generation is the real inference workload.
2. **Memory-bound ops hit a per-op floor (§T348/§B42).** Zero-copy UMA gave 0 delta
   because the floor is the round-trip latency + per-op overhead, not the copy.

Both are the same lever: **fewer per-op GPU round-trips.** The blocker is that each op's
output is downloaded to host and the next op re-uploads it, so ops cannot share a command
buffer or stay on the GPU between them.

## Decision

Introduce **device-resident tensors** — a tensor whose storage is a backend buffer that
persists across ops — and let a sequence of ops record into **one command buffer** with a
single submit/wait. An op reading a device-resident input skips the upload; an op writing
a device-resident output skips the download; only the boundaries (first upload, final
download) copy. This is the standard GPU-framework model (PyTorch device tensors).

Concretely, the pieces:
- A `Device`-tagged storage variant holding a backend buffer handle (metal `MTLBuffer`,
  vulkan `VkBuffer`) instead of a host slice. The existing resident-quant-weight
  mechanism (`vk_qweight_upload`/`mtl_qweight_upload`, §T156) is the seed.
- A backend "batch" / deferred mode: `Execute` records into a shared command buffer and
  returns a device-resident output (no wait); an explicit `Sync`/read materializes to host
  and flushes. Ops chain on-GPU with pipeline barriers (the §T343 conv pattern generalized).
- Fallback preserved: a device tensor read on the CPU/reference backend, or by an op with
  no GPU kernel, transparently downloads (§I4). `CGO_ENABLED=0` is unaffected (pure-Go
  path never creates device storage).

## Phasing (each phase is its own green fire)

- **Phase 1 — device storage primitive.** Add the `Device` storage variant + upload/
  download + `Release`, behind the backend, with round-trip tests. No op uses it yet;
  no behaviour change. Lowest risk.
- **Phase 2 — batched decode.** A decode step keeps its per-layer intermediates device-
  resident and records into one command buffer; one submit/wait per step. Measure against
  `BenchmarkGPTDecode` (target: close the ≈2.3× gap, aim for GPU-competitive decode).
  V-CROSS: greedy output identical to the current path (§T365 test generalizes).
- **Phase 3 — batched training/prefill.** Extend the deferred mode to the forward/backward
  graph (autograd tape flushes once), removing the memory-bound per-op floor (§T348).
- **Phase 4 — Vulkan parity** throughout (metal-first per §T156/the measurement host).

## Risks / gates

- L0 storage lifetime + cgo buffer ownership (like §T336's pool + §T156's resident
  buffers) — the `Release` path must free backend buffers; a device tensor outliving its
  backend is a bug (guard + test).
- The Execute contract becomes async on the batch path — the tape/decode loop must `Sync`
  before reading a value on the host. Keep the default (synchronous, per-op) path intact;
  the batch path is opt-in until proven.
- §V22: every phase measured against the real benchmark before it lands; no phase merges
  without a delta (Phase 2/3) or being a pure no-behaviour primitive (Phase 1).

## Alternatives rejected

- **Fused per-op kernels for decode** (one kernel per layer): would cut dispatches too,
  but abandons MPS and needs a bespoke fused GEMM+bias+act+attention kernel — more code,
  less general than device-resident batching, and doesn't help the memory-bound training
  ops. Device-resident tensors fix both workloads with one mechanism.
- **Zero-copy UMA alone** (ADR-0018): rejected — the copy is not the floor (§B42).
- **Routing everything small to the CPU**: rejected as a blanket default (§T361, the
  crossover is size- and hardware-dependent); `WithBackend` already gives the choice.

## Amendment: resident reductions at the consumer boundary (2026-08-18)

Device residency does not require every consumer to execute as a GPU command.
On Apple Silicon, a completed `MTLResourceStorageModeShared` buffer is coherent
UMA. If the public result is bounded while the resident tensor is large—for
example Top-K sampling over vocabulary logits—the backend should return the
bounded result without first materializing the full tensor in Go.

The first promoted instance is Metal `DeviceBuffer.TopKN`: a bounded native
heap scans the coherent resident f32 buffer and returns only k index/value
pairs. At n=32,000 and k=56 it measures 62.7-66.7 us and clears the end-to-end
gate without adding a second GPU command buffer after decode has already
synchronized. This is still a device-resident boundary: the n-element tensor
never becomes a Go tensor, and only O(k) data crosses cgo. CUDA retains its GPU
reduction because discrete-memory economics differ.

The architectural rule is capability-based, not backend-name branching. The
decoder asks whether its logits buffer implements the narrow `TopKN` contract;
unsupported backends and samplers whose semantics need arbitrary logits retain
the complete host fallback. Exact token parity and an end-to-end generation
gain are required independently of leaf timing.
