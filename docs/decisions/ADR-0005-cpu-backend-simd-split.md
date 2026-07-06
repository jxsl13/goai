# ADR-0005 — Optimized `cpu` backend; SIMD-intrinsics split (T11 / T11b)

- Status: accepted (autonomous loop, §T11)
- Date: 2026-07-05
- Relates: §I.L1b, §V3, §V9, §V5, §V11, §T11, §R1, §R3, §B2

## Context

§T11 optimizes elementwise ops. The reference backend (`ref`) must remain the
numeric truth (§V9), so optimized kernels cannot overwrite it — they need a
separate backend validated against `ref` (§V3). Two facts constrain the SIMD
work:

- `simd/archsimd` exists in Go 1.26 but is **AMD64-only** (§R1, §R3).
- The build/dev host here is **darwin/arm64**, so archsimd intrinsics cannot be
  compiled or runtime-verified locally. Merging kernels that cannot be verified
  would violate §V9/§V3 ("never merge unverified", "never skip tests").

## Decision

1. Add a separate optimized backend `backend/cpu`, registered as the preferred
   Default; `ref` stays the truth and the fallback (§I4).
2. `cpu` reaches the **verifiable Pure-Go ceiling** now: contiguous typed-slice
   loops (no per-element `Unravel`/alloc) + goroutine parallelism above a size
   threshold + compiler auto-vectorization. SIMD-class primitives live in
   `internal/simd`.
3. The **archsimd (amd64) intrinsic path is split into §T11b** — build-tagged
   `amd64 && goexperiment.simd`, with the portable `internal/simd` loop as the
   fallback everywhere. It is CI-gated (the `simd-amd64` job runs its V-CROSS),
   not host-verified, so it lands only when runtime-verifiable.

## Rationale

- On arm64 today the typed loop + auto-vectorization *is* the Pure-Go ceiling
  (archsimd is amd64-only), so §T11's "pure Go to the ceiling" is genuinely met
  here; the amd64 intrinsic is a different arch's ceiling, correctly deferred.
- Keeping `ref` as an independent slow truth is what makes §V3 CROSS meaningful.
- Elementwise ops compute the same f64 expression in `cpu` and `ref`, so their
  results are **bit-identical** → this fixes §V11 CROSSTOL for elementwise:
  tolerance 0 (reductions/GEMM get K-scaled tolerances in §T12).

## Consequences

- `Default()` prefers `cpu`; programs importing the root package get it. `ref`
  remains available and is used wherever a `cpu` kernel is absent (§I4).
- §T11b tracked in §T; §B13 records the deferral.

## Revisit if

archsimd gains ARM64 support (then the intrinsic path applies to the host too),
or a CI amd64 runner lets us promote §T11b sooner.
