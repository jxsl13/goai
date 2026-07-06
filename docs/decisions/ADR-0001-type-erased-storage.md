# ADR-0001 — Type-erased tensor storage with a runtime Dtype

- Status: accepted (autonomous loop, §T3)
- Date: 2026-07-05
- Relates: §I.L0, §C4, §V9

## Context

The Tensor element type can be f32 or f64 now, and f16/bf16/int8 later (§C4).
Two representations are possible:

1. **Generic** `Tensor[T]` — compile-time element type.
2. **Type-erased** — a single `Tensor`/`Storage` carrying a runtime `Dtype`
   tag; the backing is a typed slice held behind `any` (F32→`[]float32`,
   F64→`[]float64`).

## Decision

Type-erased storage with a runtime `Dtype` tag.

## Rationale

- **Runtime dtype is unavoidable.** Model files (safetensors/GGUF, §T19/§T22)
  carry a dtype known only at load time. A generic `Tensor[T]` forces the dtype
  into the type system, so a loader would need a type switch that reconstructs
  the generic parameter anyway — the erasure happens regardless.
- **One Backend interface over all dtypes (§I.L1).** Kernels dispatch on a
  runtime `Dtype`; a `[]Backend` registry and the autograd tape (§T13) stay
  monomorphic instead of exploding across `T`.
- **Matches proven C++ designs.** PyTorch/ATen and ggml both use type-erased
  buffers + a runtime scalar-type tag for exactly these reasons.
- **Cost is negligible.** The `any`-held slice needs one type assertion per
  whole-tensor kernel call, amortized over all elements — not per element.

## Consequences

- Less compile-time type safety at op boundaries; guarded by dtype checks +
  parity tests (§V1) and the reference backend as truth (§V9).
- Backing is `[]float32`/`[]float64` held in `any`. **Revisit** (as a §T11-class
  optimization, not correctness) whether an `unsafe`-reinterpreted `[]byte`
  buffer beats it once profiling exists — parked, not adopted.

## Revisit if

Profiling shows the `any` assertion or per-dtype slice fields dominate; or dtype
count grows enough that the tagged accessors become unwieldy.
