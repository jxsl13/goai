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

## T-01M0J9DC3ZE7RTEEHKES6R9RJ6 Eliminate ARM64 F32 MHA band Q/O copies with strided GEMM
kind: task
state: draft
created: 2026-08-21
parent: P-01M0J975XHFD5AXGP661E8G644
targets: backend/cpu/mha.go, backend/cpu/gemm_rows.go, backend/cpu/gemm_rows_arm64.go, backend/cpu/gemm_rows_amd64.go, backend/cpu/gemm_rows_default.go, backend/cpu/gemm_neon_arm64.go, backend/cpu/gemm_f32_strided_internal_test.go, backend/cpu/mha_band_rows_arm64_simd_test.go, backend/cpu/mha_test.go, backend/cpu/normattn_bench_test.go

Add lda/ldc-aware worker-callable F32 row GEMM entry points without changing the existing contiguous APIs. Implement the ARM64 SIMD path by passing independent row strides into the existing 4x16 NEON tile and handling row/column tails safely; provide amd64 SIMD and portable definitions so every build remains valid. Route only mhaFwdGemmBand through direct strided Q reads and direct strided output stores, retaining score scratch and its causal compaction. Remove qb and ob scratch from this band. Add guard-region and contiguous-equivalence tests for the strided surface plus the existing deterministic MHA digest. Pilot exact merged control versus candidate first; reject or redesign if the five forward cells do not all clear 8 percent, decode regresses, semantics drift, allocations increase, or pprof does not reduce memmove.
