# ADR-0017 — Pure-Go GEMM cache blocking parked (no delta on arm64)

- Status: Accepted (measurement-driven negative result)
- Date: 2026-07-07
- Task: §T74 (L1b-opt GEMM ladder to the Go ceiling), §R67 (BLIS/Goto packing)
- Related: §B41, §B39 (identical outcome on the Vulkan GEMM), §B27 (NEON parked
  by measurement), §T11b/§B13 (amd64-SIMD host-blocked)

## Context

The `cpu` GEMM (`backend/cpu/gemm.go`) is the ikj-order, 4-row register-blocked,
row-band-parallel kernel of §T12/§T12b. §T74 proposes climbing the classic
Go-ceiling ladder: **BLIS/Goto cache blocking (mc/kc/nc) with a packed,
cache-resident B-panel**, then wider-FMA f32 SIMD. This ADR records the outcome
of the cache-blocking rung. (The SIMD rung remains host-blocked on this arm64
box — no `archsimd` amd64 runtime for V-CROSS, cf §T11b/§B13.)

## What was built and measured

A textbook kc×nc blocked kernel: for each (jc, pc) tile, pack `B[pc:pc+kc,
jc:jc+nc]` row-major into a contiguous scratch buffer (with an f32→f64 widen on
the f32 path, hoisting the conversion out of the inner loop), then run the 4-row
register microkernel over the band reusing the resident panel. The blocking
reorders *which* (i,j) are touched when, never the per-element k-accumulation:
for a fixed (i,j) the pc-blocks run in ascending order and p ascends within each,
so the running sum is byte-for-byte the reference's ascending-p sum — **§V3/§V11
tol-0 bit-identity held green throughout** (`TestGemmCrossReferenceExact`).

A/B benchmark on the arm64 Apple M-series host, 3 samples each (`-benchtime 300ms
-count 3`), medians:

| shape (square) | unblocked (baseline) | blocked | verdict |
|----------------|----------------------|---------|---------|
| f64 512        | ~4.88 ms             | ~4.88 ms | wash |
| f64 1024       | ~37.8 ms             | ~33.1 ms | ~12 % but inside run-to-run noise, +40 % memory |
| f32 1024       | **~30.5 ms** (tight) | ~33.8 ms | **regression (~-9 %)** |

## Decision

**Discard the cache-blocked kernel; keep the unblocked register-blocked GEMM.**
Blocking does not convincingly beat the already-optimized baseline on this host —
it ties on f64-512, regresses f32-1024, and the lone marginal f64-1024 median
sits within the thermal/scheduling noise band while costing ~40 % more memory
(the packed panel + per-band scratch). Per the §C3 / V-CGO discipline, an
optimization that fails to beat the optimized pure-Go version is not merged; its
finding is parked in §B (→ §B41) and §T74 moves to parked (`~`).

## Why (root cause)

The Apple M-series memory subsystem (very high bandwidth, large shared caches)
already feeds the streaming ikj kernel close to what its scalar FMA throughput
can consume, so the GEMM is **not cache-capacity-bound at these sizes** — panel
packing adds a copy + allocation without removing a real stall. This is the same
root cause as §B39 (Vulkan register-blocked GEMM: memory-bandwidth-bound on
MoltenVK, higher arithmetic intensity didn't help) and §B27 (NEON: the loop
already runs ~1 elem/cycle). The genuine remaining GEMM headroom here is wider
SIMD lanes (f32 2×/4× FMA), which is architecturally host-blocked on arm64.

## Consequences / resume conditions

- The kernel is unchanged from §T12b; no API or numerical change; CGO0 green.
- The 1024-size benchmarks added to `gemm_test.go` are kept — useful future
  baselines regardless of this outcome.
- Resume the blocking rung only on a host whose cache hierarchy actually
  bottlenecks the unblocked stream (large-cache x86 server), **re-measuring
  before any merge** — never ship blocking on the strength of this host.
- The SIMD rung resumes with an amd64 CI runner (`goexperiment.simd` /
  `simd/archsimd`) exactly as §T11b/§B13.

## Re-measured on amd64 Zen 3 (2026-07-14) — resume condition met, still no delta

The blocking rung's resume condition ("re-measure on a large-cache x86") was met
on the Linux worker (AMD Ryzen 7 5700G, Zen 3, 16 MB L3, AVX2+FMA), on top of the
now-`f32-native` direct-store SIMD kernel (`gemmF32BandDirect`, 196 GFLOP/s — see
`SPEC-worker-linux-amd64-cuda.md` §CPU-3, ADR-0021). A packed-B variant
(`gemmF32BandPacked`: each 16-column B panel copied once into a contiguous k×16
scratch, sequential inner-loop reads, reused across the band's row-blocks;
identical f32 accumulation → identical result) was built and A/B-measured:

| F32 GEMM | direct (stride-n B) | packed-B | delta |
|----------|---------------------|----------|-------|
| 512³  | ~187 GFLOP/s | ~157 GFLOP/s | **−16%** |
| 1024³ | ~196 GFLOP/s | ~185 GFLOP/s | **−6%**  |

**REGRESSION on x86 too — DISCARDED (§C3).** The pack copy + allocation cost
exceeds any prefetch benefit: at these sizes the streaming kernel is *not*
cache-capacity- or B-read-bound (B fits L3 and is reused across the 4-row block;
the hardware prefetcher handles the stride-n loads), so packing removes no real
stall — the identical root cause as the arm64 result above and §B41/§B39/§B27.
The x86-server resume condition is now **closed with data**: blocking does not
help this kernel on this class of host either. The remaining gap to vendor SGEMM
(≈3× on this Zen 3) is microkernel saturation (register/prefetch scheduling, and
the f64-accumulation policy §V10/ADR-0021), not cache blocking.
