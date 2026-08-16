---
name: apple-gpu-rewards-parallelism-over-density
description: "On this M2, Metal kernels that raise per-thread arithmetic density lose to ones that spread work wider — textbook GPU advice inverts, but not monotonically."
metadata: 
  node_type: memory
  type: project
  originSessionId: c9844d26-52eb-4368-8358-6d744bfde57b
  modified: 2026-08-15T13:50:39.654Z
---

Textbook GPU kernel advice (register blocking, wider tiles, more accumulators per thread) is written for GPUs with deeper register files than the M2's. Here it repeatedly **loses**, and the opposite move wins — up to a crossing point.

**Measured 2026-08-15, three independent cases:**
- **GEMM tiles** (gate/up 2048x5632, M=512): BN=32 with 4 accumulators/simdgroup = 2598 GFLOP/s; BN=64 with 8 = 2444; BM=128 with 8 and 2 A-fragments = 1739; and 4x4 register blocking — the standard fix for load:compute ratio — is the **worst** at 1167 (0.63x of the shape it replaced).
- **Decode attention**: splitting `dk` across lane pairs gave **2.9x**, and across quads a further 1.05-1.18x. Both spread the same work over more threads with smaller per-thread arrays.
- **GQA staging**: consolidating 8 query heads onto one threadgroup to reuse a staged K/V tile measured **0.53-0.75x** — it traded 8x parallelism for traffic the cache was already serving free.

**But it is not a gradient to follow to the end.** BN=16 (2 accumulators) loses 14% against BN=32: an A-load plus two B-loads then amortise over only two matrix ops. The two effects cross, and the crossing point is where the tile belongs.

**How to apply:**
- When a Metal kernel is latency-bound, try *narrowing* per-thread state before widening it, and measure the neighbours on BOTH sides before calling anything an optimum.
- `maxTotalThreadsPerThreadgroup` reads register pressure directly (`mha_decode`: 1024 generic / **384** dk=64 / 1024 dk=128). But the 1024 variants reach it by SPILLING, and spilling measured 5.9-8.7x slower — so occupancy is a proxy this kernel family inverts. Never optimise it directly.
- A mechanism that explains one win may not predict the next: the pair-split `dk` win was attributed to register relief, but occupancy only moved 384→448. The real cause was coalescing (a lane pair reads two contiguous 32-dim halves), which is why the quad split then worked too.
- Related: [[gpu-kernel-runtime-loop-bounds]], [[read-the-kernel-before-claiming-a-gap]].
