---
schema: v1
prefix: SIMD
---

## SIMD-004
The SIMD kernel SHALL sit behind //go:build amd64 && goexperiment.simd with its scalar twin behind the negation, so exactly one definition exists per build.

Rationale: The default CGO_ENABLED=0 build stays unchanged. The CI SIMD gate runs GOEXPERIMENT=simd go build -tags=simd ./..., which builds but does not test the experiment. Migrated from the linux-amd64-cuda worker spec Iw1.

## SIMD-005
The archsimd intrinsic SHALL be gated at runtime by archsimd.X86.AVX() or .FMA(), so a binary built with the experiment falls back to scalar on a pre-AVX CPU instead of raising an illegal instruction.

Rationale: Go 1.26 simd/archsimd provides Float32x8 (8 lanes) and Float64x4 (4 lanes). Runtime feature detection is the same invariant the accelerator backends follow. Migrated from the worker spec Iw2.

## SIMD-006
WHERE the amd64 SIMD build, the bit-exact SIMD kernels SHALL vectorize the free dimension j rather than the reduction k, and use separate Mul and Add rather than MulAdd.

Rationale: Vectorizing the reduction changes summation order. On amd64 the scalar twin rounds twice (mul then add) while FMA rounds once, so a bit-exact kernel must NOT fuse. This is architecture-specific: see the arm64 companion rule, where the rule inverts. Migrated from the linux-amd64-cuda worker spec Iw5, which was written for that host only.

## SIMD-007
WHERE the arm64 NEON build, the bit-exact SIMD kernels SHALL use FMLA rather than separate FMUL and FADD, since the Go arm64 backend already contracts the scalar twin into FMADDS.

Rationale: The rule inverts against amd64 and following the amd64 form here would BREAK bit-exactness, not preserve it: the repo verified on objdump that the scalar SAXPY loop compiles to scalar FMADDS (backend/cpu/gemm_neon_arm64.go), and the NEON kernel header records that each C element accumulates its k products in ascending p order in one fused-FMA chain (gemm_neon_arm64.s). A kernel using separate mul and add would round twice where the scalar rounds once, failing TestGemmCrossReferenceExact. Note also that the real arm64 NEON kernels live in backend/cpu, not in internal/simd, which has no arm64 files at all.

## HOISTING-A-SLICE-CUT-CAN-COST-MORE-THAN-REPEATING-IT-001
IF a loop-invariant slice cut is hoisted out of an enclosing loop, THEN the implementing agent SHALL measure it, because extending the slice live range across that loop can cost more than recomputing the cut.

Rationale: In the SSM scalar scan the bs and cs cuts depend only on t, not on d, so recomputing them inside the d loop repeats identical work D times per t and the profile attributed about 0.5 of 6.5 seconds to that setup. Hoisting them above the d loop REGRESSED the benchmark: plus 1.12, plus 0.72 and plus 1.06 percent across three of four cells, geomean plus 0.90. A second variant that also dropped the redundant re-slice was worse still at plus 3.05 percent on the largest shape. The likely cause is live range: two slice headers kept alive across the whole d loop cost more in register pressure than the cut costs to redo, and a cut local to the loop body stays in registers. This is the same shape as the finding that redundant arithmetic inside a latency shadow is free — profile attribution on setup lines suggested a saving that restructuring could not collect. Keep the cuts adjacent to the loop that uses them unless a measurement says otherwise.
