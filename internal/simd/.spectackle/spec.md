---
schema: v1
prefix: SIMD
---

## SIMD-001
The SIMD kernel SHALL sit behind //go:build amd64 && goexperiment.simd with its scalar twin behind the negation, so exactly one definition exists per build.

Rationale: The default CGO_ENABLED=0 build stays unchanged. The CI SIMD gate runs GOEXPERIMENT=simd go build -tags=simd ./..., which builds but does not test the experiment. Migrated from the linux-amd64-cuda worker spec Iw1.

## SIMD-002
The archsimd intrinsic SHALL be gated at runtime by archsimd.X86.AVX() or .FMA(), so a binary built with the experiment falls back to scalar on a pre-AVX CPU instead of raising an illegal instruction.

Rationale: Go 1.26 simd/archsimd provides Float32x8 (8 lanes) and Float64x4 (4 lanes). Runtime feature detection is the same invariant the accelerator backends follow. Migrated from the worker spec Iw2.

## SIMD-003
The a bit-exact SIMD kernel SHALL vectorize the free dimension j rather than the reduction k, and use separate Mul and Add rather than MulAdd.

Rationale: Vectorizing the reduction changes summation order; FMA fuses to one rounding where the scalar twin has two. Both break bit-exactness against the scalar reference. Load and store of the accumulator preserves the += contract, which convolution depends on through the shared banded GEMM. Migrated from the worker spec Iw5.
