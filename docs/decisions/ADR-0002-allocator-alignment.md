# ADR-0002 — Allocator abstraction; alignment is advisory in L0

- Status: accepted (autonomous loop, §T4)
- Date: 2026-07-05
- Relates: §I.L0, §T4, §T11, review NOTE (§B10-adjacent), ADR-0001

## Context

§T4 asks for a Device + Allocator with alignment "as a parameter, not a
SIMD-coupled assumption" (review-hardened). Storage holds typed slices
(`[]float32`/`[]float64`) per ADR-0001, allocated with `make`. Go guarantees
alignment only to the element type (4B for f32, 8B for f64) — arbitrary
over-alignment (e.g. 32B/64B for AVX2/AVX-512) is impossible for a typed slice
without `unsafe` byte reinterpretation.

## Decision

1. `Allocator` is an interface: `Alloc(dtype, n) any` / `Free(any)`. Two impls:
   - `Heap` — `make`-based, `Free` is a no-op. Default, always correct.
   - `Pool` — sync-pooled per (dtype, power-of-two size class); reused buffers
     are zeroed on Alloc to preserve the zeroed-memory contract.
2. Alignment is an **advisory** allocator parameter: recorded and queryable, but
   L0 honors only natural (element-type) alignment. No `unsafe` in L0 (§I3).

## Rationale

- Keeps L0 `unsafe`-free and correct (ADR-0001), while giving ops a real
  allocation-reuse path to cut GC pressure in hot loops.
- Guaranteed over-alignment belongs where it pays off and can be measured: the
  SIMD task (§T11), which may introduce an `unsafe`-backed byte arena under its
  own ADR. Putting it in L0 now would be premature optimization (review §4).

## Consequences

- SIMD kernels that need 32B/64B alignment must not assume L0 provides it; they
  request it explicitly at T11. Tracked as §B11.
- `Pool` reuse is bounded by power-of-two size classes (some slack memory).

## Revisit if

T11 benchmarks show natural alignment materially hurts SIMD throughput → promote
a byte-arena allocator with guaranteed alignment (new ADR).
