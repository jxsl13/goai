---
schema: v1
---

## T-01KYJR5YCJF4M9BC6F960CGG9Z Investigate the worker-pool park/wake cost that dominates small-model training steps
kind: task
state: draft
created: 2026-07-27

FLAGGED UPWARD from the nn sweep — the finding is in backend/cpu but was found while profiling nn, and if it holds it is worth more than every nn optimization combined for small-model training.

OBSERVATION: pprof over BenchmarkTrainStepAdamF64 (193,592 ns/op, 296,828 B/op, 168 allocs) is dominated by worker-pool park and wake, NOT by any kernel and not by nn at all — runtime.usleep 36.2%, pthread_cond_wait 14.9%, kevent 13.1%, cpu.poolWorker 48.9% cumulative, with total samples at 274% of wall-clock (i.e. cores are spinning, not computing). The actual kernels are gemmF32Band at 2.2% and gemmF64Band at 1.4%.

SITE: backend/cpu/cpu.go:245 parallelWork and the pool's spin/park policy.

WHY THIS MATTERS: at small model sizes the per-dispatch fork/join cost exceeds the work dispatched. Every op in a small training step pays it, and it compounds with any change that increases dispatch count. It also interacts with the Muon task in this same round: routing Newton-Schulz through the parallel GEMM adds roughly 30 fork/joins per step, so if the park/wake cost is as large as this profile suggests, that task's step-4 estimate is optimistic and the two must be measured together.

SCOPE THIS AS AN INVESTIGATION, NOT A PRESCRIBED FIX. The profile is strong evidence that something is wrong, but the shape of the fix (spin-then-park thresholds, a serial cutoff below some work size, batching dispatches, or a persistent worker handoff) depends on measurements this sweep did not make. Concretely: (1) determine the minimum work size at which parallel dispatch beats serial on this host, per dtype; (2) check whether a serial fast path already exists and at what threshold (gemm.go:130 has a serial small path — establish whether its cutoff is calibrated or arbitrary); (3) measure fork/join latency in isolation; (4) only then propose a change.

VALIDATION GATE (benchmark only): BenchmarkTrainStepAdamF64 (nn/train_bench_test.go) as the end-to-end signal, plus a new microbenchmark that dispatches a trivially small parallel op in a loop to measure fork/join cost directly, and the existing gemm_grind_bench_test.go direct-kernel benchmarks at SMALL shapes (the current set starts at 512, which is far above where this effect lives). Sweep shapes down to 32 and 64 — the crossover is what matters, not the large-shape numbers.

EXPECTED: unknown, deliberately. The profile suggests a large fraction of a small training step, but a profile that shows spinning does not by itself prove the spinning is removable — some of it may be unavoidable synchronization that would simply move. State the measured crossover before claiming a win.

BIT-IDENTITY BAR: any change to parallel decomposition must preserve per-output reduction order. The band kernels currently guarantee each C element accumulates its k products in ascending order in one chain; a change that alters banding or work-splitting could break the tolerance-0 cross-reference gate (TestGemmCrossReferenceExact, TestConvCrossReferenceExact). A change that only alters WHEN workers park, not how work is split, is bit-identical by construction — prefer that class.

COORDINATION NOTE: a separate agent was researching the backend package concurrently in this round. Check its findings before starting, to avoid duplicate or conflicting work on the same file.

## P-01M0J5GFQJET5VFRAN479DTRES Align ARM64 F32 MHA forward bands to the four-row NEON tile
kind: proposal
state: active
created: 2026-08-21
grilled: 2026-08-21 open=0
targets: go:cpu.mhaFwdBandRows, go:cpu.mhaFwdGemmF32, go:cpu.mhaFwdGemmBand, go:cpu.gemmF32BandNeonCols, go:cpu.gemmF32RowsScalar, go:cpu.BenchmarkMHA512

Exact base is verified merge 305dd29b65cccf0521bded9f773546b3e587c166. On physical Apple M2 Pro with Go 1.26.6 and GOEXPERIMENT=simd, BenchmarkMHA512/fwd/cpu has a count-seven median of 1.096232 ms at GOMAXPROCS=10. A 5 s CPU profile attributes 6.10 sampled seconds to gemmF32RowsScalar and 6.62 to gemmF32Tile4x16Neon across workers. The dynamic forward scheduler uses 30-row bands even though both QK and PV ARM64 GEMMs consume four-row NEON tiles, forcing two rows of every full band plus the final residue through the scalar fallback. Sweep ARM64 SIMD band sizes divisible by four around the current load-balancing point, retain the smallest robust winner, and keep default, non-ARM64, F64, backward, and decode paths unchanged. Gate the change with three paired count-seven M2 campaigns at 500 ms over causal MHA512 plus full-attention and GQA controls. Require at least 1.15x complete-operation median speedup in the causal primary cell in every campaign, no statistically significant regression in controls, unchanged numerical output under existing CPU-002 parity, no extra steady-state allocations, and a refreshed profile showing the avoidable full-band scalar tail removed. Pin exact control and candidate commits and report the tile-grain scheduling pattern to perfscan.

## T-01M0J5KPW5EFY8TM91JY3WZFD2 Sweep ARM64 four-row-aligned MHA forward bands
kind: task
state: draft
created: 2026-08-21
parent: P-01M0J5GFQJET5VFRAN479DTRES
targets: backend/cpu/mha.go, backend/cpu/mha_band_rows_arm64_simd.go, backend/cpu/mha_band_rows_default.go, backend/cpu/mha_band_rows_arm64_simd_test.go, backend/cpu/mha_band_rows_default_test.go, go:cpu.mhaFwdGemmF32, go:cpu.mhaFwdGemmBand, go:cpu.gemmF32BandNeonCols, go:cpu.gemmF32RowsScalar, go:cpu_test.BenchmarkMHA512, go:cpu_test.BenchmarkAttention

Base 305dd29b65cccf0521bded9f773546b3e587c166 uses one 30-row scheduler grain for both architectures. That grain is correctly divisible by the amd64 six-row AVX2 tile but not by the Apple arm64 four-row NEON tile, so a global replacement would trade one architecture for another. Sweep ARM64 SIMD candidates 24, 28, 32, 36, and 40 against control 30 using same-binary-or-exact-commit paired M2 measurements. Promote only the robust winner through build-tag-specific constants: arm64 plus goexperiment.simd receives the selected multiple of four; every other build retains exactly 30. Add direct build-scoped assertions for the four-row divisibility and the untouched default value. Measure causal BenchmarkMHA512 plus seq128 and seq512 full and GQA forward controls and the decode control. Three paired count-seven 500 ms campaigns must show at least 1.15x primary speedup in every campaign, no statistically significant regression in any forward control, unchanged decode, no extra steady-state allocations, and existing MHA parity plus default/SIMD/race/cross-build suites green. Refresh the CPU profile to verify scalar full-band remainder work disappears rather than merely moving time. Pin exact commits and report the architecture tile-grain scheduling pattern to perfscan.
