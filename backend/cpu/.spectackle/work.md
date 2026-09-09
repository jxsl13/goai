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

## R-01M23Q8QJDE3Y878N2CZBNPAWJ Diagnose repeated parallel control allocation-byte movement without rescoring rejected GELU V2
kind: research
state: active
created: 2026-09-09
parent: P-01M23BZPZNF9VV3FF326AQX28E
refs: R-01M23B9YZ1E16ANNE778WX5G7T
targets: backend/cpu/bench_test.go

READ-ONLY bounded research after rejected V2 T-01M23JPZT8FDM. Existing full campaigns cannot be resampled/rescored and immutable gelu-old.test/V1/gelu-v2.test and shared harness stay unchanged. First server research pack did not identify cause. Independent audit all24largepublictargets2.058–3.924x and48smallcells pass, but parallel Softplus B/op medians2097482→2097484,2097481→2097484,2097482→2097484 repeat3/3 despite unchangedallocs and nonsignificant per-campaign p. SiLUBackward2/3 +4,+1; noGELU B/op increase. Determine whether source/runtime accounting evidence explains these bytes or shows actual candidate allocation path changes. Research ONLY: no code/lifecycle writes, commits/pushes or benchmark/test/profile/build launches. Read .claude/commands/spectackle.md, get this item, then use find/get code nodes before known-file reads. Inspect raw controls and adaptively selected N; Go1.27.1 local SDK testing/benchmark.go and runtime allocation accounting; compare old dd1e779e and V2 runtime361955ac source and emitted relevant functions if useful. Get exact allocation/call path of BenchmarkSoftplusF64_256K_cpu via benchOn/public Execute/unary parallel scheduler; include Sigmoid/SiLU differences. Identify deterministic source deltas vs probabilistic process-wide effects; do NOT assert causality from nonsignificance or manufacture threshold. Draft minimal predeclared diagnostic protocol with old-old negative control, equal fixed iteration counts, raw total bytes/allocations if needed, exact source/binary pins, serialized heavy runs, independent verify and no original gate relaxation. Recommend evidence needed before any V3 implementation or benchmark-contract revision, which needs a new reviewed task. Use fresh cheap researcher. Output report via apply_patch /private/tmp/goai-m2-f64-gelu-neon-26BhJN/v2-allocation-research.txt with exact sources/commands/results/uncertainties. Root owns evidence/docs/lifecycle; generalizable substantive finding must be filed/deduped in perfscan. No external/model leadership claim.
