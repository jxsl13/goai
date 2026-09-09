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

## P-01M237J19FFE5SYSTMV3442H6H Measure direct-call scalar F64 GELU backward before a wider SIMD redesign
kind: proposal
state: draft
created: 2026-09-09
refs: R-01M236QP4YEFDABMQ07MC43EHH
grilled: 2026-09-09 open=0
targets: go:cpu.activationBackwardF64, go:cpu.geluBackwardF64KernelCPU, go:cpu.geluGradF64, go:cpu_test.TestActivationBackwardF64CPUMatchesRef

Consumes research R-01M236QP4YEFD. Baseline source137b3355 (runtime identical to merged06993c9e), Go1.27.1 CGO0 ARM64 SIMD on Apple M2 Pro, frozen binary fe3678cb26abf688919cb0f3ed8773bf3f05dce54bed8c70618a4de0a9fbcf09. Nine unchanged baseline samples: production GELU backward256K median3954529ns at GOMAXPROCS1 and800383ns at12; forward256x2048 median5267461ns and961250ns. Baseline only, no A/B or incumbent claim. A separate5s backward profile reports math.erf35.01% flat, math.archExp24.74%, activationBackwardF64.func1 11.32%, geluGradF64 4.61%; profile is diagnostic, not benchmark evidence. First bounded experiment replaces function-value grad dispatch inside the scalar production GELU backward loop with direct calls to the unchanged geluGradF64, preserving operation order, exact-erf definition, SIMD route, validation/reference fallback, Contiguous conversions, allocation ownership, and parallel policy. DEVIRTUALIZING-REMOVES-AN-FMA-BARRIER-001 and PROC-009 require proving bit identity and a one-ulp mutation oracle; do not duplicate/reassociate arithmetic or relax any tolerance. Add exact edge/shape/strided/parallel tests and a same-harness benchmark matrix, freeze baseline before runtime edits, alternate old/new binaries at least nine pairs with warmups excluded and fixed compiler/build/data. Candidate accepted only for repeatable production leverage without meaningful small/parallel regressions or allocation regressions; noise cannot be called a win. Revert runtime candidate and retain rejection evidence if gate fails. No ARM64 erf approximation, dispatch registry redesign, forward change, Metal change, or package floor change. All durable changes follow independent plan/verification review, perfscan reporting of generalizable findings, PR, all CI jobs success, then authorized merge and remote branch deletion. Existing Spectackle global-context debt is tracked separately in issue284 and must not be hidden.

## T-01M237QTE7E8ETPPC9MGEQVXGF Test and measure exact scalar GELU backward callback specialization
kind: task
state: draft
created: 2026-09-09
parent: P-01M237J19FFE5SYSTMV3442H6H
refs: R-01M236QP4YEFDABMQ07MC43EHH
targets: go:cpu.activationBackwardF64, go:cpu.geluBackwardF64KernelCPU, go:cpu_test.TestActivationBackwardF64CPUMatchesRef, backend/cpu/gelu_bwd_direct_test.go

SCOPE: worktree /private/tmp/goai-m2-f64-gelu-5cRcwS/repo only, inherited runtime source137b3355 identical to merged06993c9e. Runtime file backend/cpu/activation_bwd_f64.go; new tests and common benchmark harness backend/cpu/gelu_bwd_direct_test.go in package cpu_test. Orchestrator owns Spectackle, evidence docs and benchmark runner, GitHub and merges; implementer owns only these two Go files and temporary binaries under /private/tmp/goai-m2-f64-gelu-5cRcwS. Do not modify reference, math formula, SIMD files, tolerance policy, scheduling, other operations, API or toolchain. BASELINE FIRST: add tests and end-to-end BenchmarkGELUBackwardF64DirectDispatch subbenchmarks n2048 and n262144 using backend.Execute, CPU registration, exact same deterministic bench.RandF64 seeds1/2, output allocation included, ReportAllocs. Freeze old test binary with Go1.27.1 CGO_ENABLED0 GOEXPERIMENTsimd GOTOOLCHAINlocal and retain source hash/build metadata before runtime edits; run old focused tests and mutation. MUTATION: temporarily change geluGradF64 return by one ULP, run existing TestActivationBackwardF64CPUMatchesRef and TestCPUGeluBackwardCrossReference; demonstrate failure from the numerical oracle, restore mutation. If mutation survives, prove target reachability with temporary panic and repair oracle before rewrite. IMPLEMENT: specialize activationBackwardF64 into geluBackwardF64ScalarKernelCPU, remove op/grad parameters, call unchanged geluGradF64 directly inside unchanged parallel body, use fixed OpGELUBackward fallback, update its sole caller. Preserve original vexpF64Fast branch verbatim and every eligibility guard, both Contiguous calls, NewOn, recorder-free reference fallback and input/output ownership. Keep geluGradF64 body/constants exactly unchanged; inspect generated code to report whether direct calls survive and ensure no new arithmetic fusion. CORRECTNESS: add small/ragged/parallel lengths including2048,262144,200003, signedzeros, finite huge/tiny values, NaN/Inf classes, empty/rankzero, offset/transposed views and input immutability; compare finite/signedzero result bits against reference for default/ARM64 scalar path without loosening existing AMD64 tolerant route. Preserve fallback behavior and validation errors. VERIFY: GOMAXPROCS1 and12, default and SIMD, Go1.27.1 CGO0 go test ./backend/cpu -run "Test(ActivationBackwardF64CPUMatchesRef|CPUGeluBackwardCrossReference|GELUBackwardF64Direct)" -count1 -timeout1800s; all backend/cpu tests default and SIMD; pure-Go full build; AMD64 SIMD test-binary compilation; race focused tests when supported. Use actual Go flag spelling with spaces. PERFORMANCE: after binaries frozen, orchestrator runs three alternating count-seven campaigns of the common2048/262144 harness at GOMAXPROCS1/12, discarded warmups, benchtime1s, no concurrent builds/tests. SCALAR-F64-GELU-DIRECT-PERF-001 requires target262144/GMP1 >=1.05x and p<0.05 every campaign; CONTROLS001 rejects reproducible >3% regression or allocation increase. Small effects must exceed within-arm noise under PROC-INTERLEAVE001. No kernel speedup or incumbent claim until verified; no runtime promotion if gate fails. HANDOFF: report precise commands/status, artifact paths and hashes, mutation evidence, full diff and remaining gates; do not commit/archive/push. An independent fresh-context verifier must rerun from diff, record validate verdict before archival, and all16 CI jobs must pass before authorized PR merge. R-01M236QP4YEFD baseline/profile are diagnostic context, not old/new performance proof.
