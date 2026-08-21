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

## ARM64-F64-EXP-PERF-001 {applies: go:simd.expNegPairsNeonF64,asm:simd.expNegPairsNeonF64,go:simd.ExpSumF64,go:simd.ExpScaledF64,go:simd.SigmoidF64,go:simd.SoftplusNegLLSumF64}
WHEN three paired count-seven M2 campaigns measure the five declared F64 cells, the arm64 SIMD F64 path SHALL retain only when the direct leaf reaches 2.00x control, every complete-operation median reaches 1.25x control, and every target has zero allocations.

## ARM64-F64-EXP-NUMERIC-001 {applies: go:simd.expNegPairsNeonF64,asm:simd.expNegPairsNeonF64,go:simd.ExpSumF64,go:simd.ExpScaledF64,go:simd.SigmoidF64,go:simd.SoftplusNegLLSumF64}
WHEN the arm64 SIMD candidate evaluates any declared F64 operation, the result SHALL match the scalar reference within 1e-13 relative error and preserve scalar NaN, infinity, signed-zero, and exact deep-underflow behavior.

## ARM64-F64-EXP-MEMORY-001 {applies: go:simd.ExpSumF64,go:simd.ExpScaledF64,go:simd.SigmoidF64,go:simd.SoftplusNegLLSumF64}
WHEN a declared F64 operation receives odd lengths, aliased ExpSum storage, or distinct input and output buffers, the implementation SHALL use a length-determined scalar tail, preserve in-place ExpSum safety, and leave distinct input buffers unchanged.

## ARM64-F64-EXP-FALLBACK-001 {applies: go:simd.ExpSumF64,go:simd.ExpScaledF64,go:simd.SigmoidF64,go:simd.SoftplusNegLLSumF64,go:simd.ExpSumF64~2,go:simd.ExpScaledF64~2,go:simd.SigmoidF64~2,go:simd.SoftplusNegLLSumF64~2}
WHEN an input lane is outside the vector polynomial safe domain or the build is not arm64 with goexperiment.simd, the implementation SHALL preserve scalar API semantics without imposing a new input restriction and leave non-target builds unchanged.

## intent
- T-01KYJPYBM5E7YAG33QW56DWEVW Build the f64 NEON transcendental leaf so nine ops stop falling to scalar math.Exp on arm64: Archived after PR #1127 head e0b7095dfa176a3fefa8b14a5eca0a8261a7d498 completed the full 15-check CI matrix successfully (run 32468409469). The final implementation adds an Apple arm64 goexperiment.simd two-lane F64 NEON exponential leaf and composes ExpSumF64, ExpScaledF64, SigmoidF64, and SoftplusNegLLSumF64 with scalar fallback for unsafe, non-finite, and subnormal-boundary domains; odd tails, [body truncated at tombstone retention cap]
- P-01M0HWDB7BEBY99M96950C5DHE Apple arm64 fused F64 SSM selective scan: Single-task proposal completed by archived task T-01M0HWBG9QEC2 and GoAI PR #1128. The Apple arm64 F64 SSM recurrence now has a fused NEON fast path, numeric-domain proof, scalar fallback, exact range semantics, three statistically significant physical M2 Pro benchmark campaigns, and complete local plus hosted CI evidence. Product gains are internal geomeans -79.08% to -84.45% and backend/cpu end- [body truncated at tombstone retention cap]

## ARM64-F64-SSM-PERF-001 {applies: go:simd.ssmChannelNegNeonF64,asm:simd.ssmChannelNegNeonF64,go:simd.SSMScanF64~3}
WHEN paired count-seven M2 campaigns measure internal 512x2048x16, the arm64 SIMD SSM path SHALL retain only with at least 20 percent lower median latency, p below 0.05, and zero allocations.

## ARM64-F64-SSM-NUMERIC-001 {applies: go:simd.ssmChannelNegNeonF64,go:simd.SSMScanF64~3,go:simd.TestSSMScanF64Accuracy}
WHEN valid decay-domain fixtures use N 1, 2, 3, 16, 17, or 128 with or without D-skip, the fused arm64 F64 SSM implementation SHALL match scalar output and recurrent state within 1e-10 relative error.

## ARM64-F64-SSM-FALLBACK-001 {applies: go:simd.ssmNeonRangeSafe,go:simd.SSMScanF64~3,go:simd.SSMScanRangeF64~3,go:simd.TestSSMScanF64UnsafeDomainFallsBackBeforeMutation}
WHEN delta or A violates sign, finiteness, or the proven product range, the arm64 SIMD F64 SSM dispatcher SHALL select the scalar scan before mutating output or recurrent state and preserve API semantics.

## ARM64-F64-SSM-MEMORY-001 {applies: go:simd.ssmChannelNegNeonF64,go:simd.BenchmarkSSMScan_SIMD_512x2048x16,go:simd.BenchmarkSSMScan_SIMD_512x2048x128}
WHEN the fused path processes any declared N shape including an odd state tail, the optimized arm64 F64 SSM scan SHALL use no heap scratch and report exactly 0 B/op and 0 allocs/op in the internal benchmark.

## ARM64-F64-SSM-ARCH-001 {applies: go:simd.ssmChannelNegNeonF64,asm:simd.ssmChannelNegNeonF64}
WHEN the arm64 goexperiment.simd test binary is inspected, the fused arm64 F64 SSM leaf SHALL contain two-lane D2 vector state arithmetic together with FRINTN and exponent-bit construction from the proven degree-13 negative-exp approximation.

## ARM64-F64-SSM-SCOPE-001 {applies: go:simd.SSMScanF64~3,go:simd.SSMScanRangeF64~3,go:simd.ssmChannelNegNeonF64}
The F64 SSM optimization SHALL change only arm64 with goexperiment.simd product code and leave WKV, generic Exp capability flags, non-target product implementations, and backend ownership allocations outside this task.

## ARM64-F64-SSM-E2E-PERF-001 {applies: go:cpu_test.BenchmarkSSMF64_512x1024x16_cpu,go:cpu.ssmKernelCPU,go:cpu.ssmParallelScanF64,go:simd.SSMScanRangeF64~3}
WHEN paired count-seven M2 campaigns measure backend/cpu 512x1024x16, the arm64 CPU SSM path SHALL retain only with at least 15 percent lower median latency and p below 0.05.

## ARM64-F64-SSM-RANGE-001 {applies: go:simd.SSMScanF64~3,go:simd.SSMScanRangeF64~3,go:simd.TestSSMScanRangeF64BitExactVsWhole}
WHEN the same valid fixture runs through whole and range entry points, the fused arm64 F64 SSM implementation SHALL pass go:simd.TestSSMScanRangeF64BitExactVsWhole with bit-identical output and recurrent state.

## ARM64-F64-WKV-PERF-001
WHEN three alternating count-seven M2 campaigns measure BenchmarkWKVScan_SIMD_512x1024, the arm64 SIMD WKV path SHALL retain only with at least 35 percent lower median latency, p below 0.05, and exactly 0 allocs/op.

## ARM64-F64-WKV-E2E-PERF-001
WHEN three alternating count-seven M2 campaigns measure backend/cpu BenchmarkWKV_512x1024, the arm64 CPU WKV path SHALL retain only with at least 20 percent lower median latency and p below 0.05.

## ARM64-F64-WKV-NUMERIC-001
WHEN RWKV-like and boundary fixtures exercise fresh or continuing WKV scans, the arm64 SIMD WKV implementation SHALL match scalar output and AA/BB/PP state within 1e-10 relative error.

## ARM64-F64-WKV-FALLBACK-001
WHEN a vector pair encounters non-finite operands or exponential arguments below the proven safe domain, the arm64 WKV dispatcher SHALL rerun that pair through wkvScanStateScalar from unchanged state and overwrite partial output.

## ARM64-F64-WKV-STATE-001
WHEN identical tokens run whole or in uneven chunks with carried AA/BB/PP, the TestWKVScanStateF64ChunkEqualsWhole SHALL report bit-identical output and final recurrent state.

## ARM64-F64-WKV-RANGE-001
WHEN two-aligned ranges cover the same channels as one whole scan, the TestWKVScanRangeF64BitExactVsWhole SHALL report bit-identical output for every declared shape.

## ARM64-F64-WKV-MEMORY-001
WHEN the fused path processes paired channels or a scalar channel tail, the BenchmarkWKVScan_SIMD_512x1024 SHALL report exactly 0 B/op and 0 allocs/op without heap scratch.

## ARM64-F64-WKV-ARCH-001
WHEN the arm64 goexperiment.simd test binary is inspected, the fused WKV leaf SHALL contain D2 recurrence arithmetic, FRINTN range reduction, and exponent-bit construction.
