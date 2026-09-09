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

## T-01M0JAQS4XE65VZ52P8SZTGDC6 Preflight MHA-shaped Accelerate versus NEON head GEMMs on M2
kind: task
state: done
created: 2026-08-21
parent: P-01M0JAMADPFG5R8S5TX1BDAB7F
grilled: 2026-08-21 open=0
targets: backend/cpu/gemm_amx_bench_test.go

Add benchmark-only cells for score shapes 128x64x128 and 512x64x512 plus output shapes 128x128x64 and 512x512x64 to the existing ADR-0027 path harness. Measure NEON and Accelerate in alternating count-seven physical-M2 campaigns from one exact binary. Advance to stride-aware binding and full MHA only if Accelerate is at least 1.35x faster in every head GEMM cell, providing margin for per-head cgo calls and causal overcompute; otherwise reject the proposal without production changes.

## R-01M236QP4YEFDABMQ07MC43EHH Establish the M2 F64 GELU vectorization gate after the Go 1.27.1 rebuild
kind: research
state: draft
created: 2026-09-09
targets: go:cpu.vgeluF64~2, go:cpu.vgeluGradF64~2, go:cpu.vgeluF64, go:cpu.TestVGeluF64Accuracy

Current source137b3355 retains vexpF64Fast=false on ARM64; GELU forward/backward execute scalar math.Erf/Exp while AMD64 has Cephes erfF64x4/expF64x4 plus scalar bit-twins. This is the next remaining composite after merged ARM64 F64 Softplus. Research only: inspect existing numerical contracts, freeze Go1.27.1 ARM64 SIMD CPU benchmark binary, measure existing production F64 GELU forward256x2048 and backward256K at GOMAXPROCS1/12 as an initial baseline. No implementation or speedup claim until a dedicated capability, scalar-tail identity, finite and special-value accuracy, NaN/signedzero semantics, real caller reachability, independently verified interleaved old/new benchmarks and same-semantics incumbent comparison are established. Exact-erf GELU definition per archived ADR0004 remains binding; tanh approximate GELU is not a substitute. The existing ARM64 backward CPU/ref comparison is bit-exact, whereas AMD64 SIMD has an explicit tolerant route: any proposed ARM64 relaxation requires an explicit new measured contract and keeps default exact coverage. Primary local references: pinned Go1.27.1 src/math/erf.go; backend/cpu/vexp_amd64.go erfF64x4, erfF64poly, geluF64poly; backend/cpu/vgelu_f64_internal_test.go; activation_bwd_f64_test.go. This item will be consumed by a bounded implementation proposal or closed with evidence; current PR1247 is separate and remains awaiting finalCI. Report generalizable confirmed gain upstream to perfscan per standing user mandate.
