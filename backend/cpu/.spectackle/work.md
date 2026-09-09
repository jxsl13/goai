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

## T-01M23JPZT8FDM9F9P69HMMYGFH Skip unused GELU erf arms only for uniform-small ARM64 vectors and requalify
kind: task
state: active
created: 2026-09-09
parent: P-01M23BZPZNF9VV3FF326AQX28E
refs: R-01M23B9YZ1E16ANNE778WX5G7T
grilled: 2026-09-09 open=1
targets: backend/cpu/vgelu_f64_arm64.go, backend/cpu/vgelu_f64_neon_test.go

Bounded V2 follow-up under approved active parent P-01M23BZPZNF9V. V1 T-01M23C5EWPE8P was numerically correct but rejected: all24 large target cells passed, small n2048 active-forward Execute regressed68.95–76.65% in all3campaigns/GMP1,12; independent evidence at internal/benchcompare/leadership/evidence/m2-f64-gelu-neon-20260909. Research R-01M23B9YZ1E16 and perfscan966. No external leadership claim.

OWNERSHIP: fresh implementer owns ONLY backend/cpu/vgelu_f64_arm64.go and backend/cpu/vgelu_f64_neon_test.go on /private/tmp/goai-m2-f64-gelu-neon-26BhJN/repo. Root owns lifecycle, docs/evidence, commits/push/PR and benchmarks; no assembly or unrelated route changes. Read .claude/commands/spectackle.md fully and task/parent via CLI. Use apply_patch for every local edit. Never read/edit .spectackle bytes. No mutations, commits, lifecycle writes or benchmarks without root coordination; PROC-MUTATION-COMMIT-SERIAL-001 forbids any committing activity while mutation live. Claim work/lease if available; known serving-root-only work start fails to land from main: use assigned isolated branch, report instead of creating another repository.

IMPLEMENT: after existing erfSmall evaluation and before exp/P/Q in erfF64x2GELU, compute small := ay.Less(geluNVOne), lanes := small.ToInt64x2(); if lanes.GetElem(0)!=0 && lanes.GetElem(1)!=0 return erfSmall. Reuse small in final IfElse. ARM64 SDK ops_arm64.go provides Mask64x2.ToInt64x2 (true allbitsset) and constant-index Int64x2.GetElem; do not invent Mask.All. Use only both-small strict abs(y)<1 fast return; mixed/one-small, abs(y)==1 boundaries and big pairs retain V1 complete path. Preserve existing coefficient bytes, operation ordering, two independently rounded backward exp arguments, scalar twins, exact fallback and aliases, dedicated gate/global vexpF64Fast=false, default/AMD64/F32/ref semantics. Existing ARM64-F64-GELU-NUMERIC,DOMAIN,FALLBACK,EXP-ORDER,SCOPE,PERF,CONTROLS rules remain unchanged. No polynomial/assembly/one-exp redesign or eligibility expansion. Only fast-path selection permitted.

TEST-FIRST: add explicit direct helper tests comparing both outputs bitwise with existing scalar operation-order twin for pair permutations: both-small with opposite signs and signedzero/subnormals; small/middle and middle/small; small/big and big/small; mixed sign; predecessor/exact/successor ±1 and ±6. Include public/direct forward/backward body-tail tests on corresponding x boundaries as needed; all old dense/random/range-reduction/numeric/whole-wrapper fallback/alias/view/metadata/Recorder/gate reachability tests remain active. New tests must pass V1 before runtime edit. Keep shared untagged vgelu_f64_bench_test.go SHA b53599699510a31f3a2d086f08f1b11b968b9ad1f25941e75c8a1f62fc49c9f3 byte-identical; never overwrite frozen gelu-old.test or V1 gelu-new.test.

MUTATION: after root grants an exclusive mutation window, add a compilable incorrect output ONLY to the new both-small return (e.g. erfSmall.Add(BroadcastFloat64x2(1e-8))). Run focused new test and require actual assertion failure (not compiler failure). Optionally mutate && to || to prove mixed pairs cannot take early return. Restore exact source bytes with apply_patch, verify sourceSHA and focused PASS; notify root the window is closed. All attempts retained even invalid ones. No lifecycle operations or commits during mutation. Do not silently mark success if command fails before tests.

VERIFY: pinned /private/tmp/goai-go1271-j1cLvq/go/bin/go, PATH matchesSDK, GOCACHE=/private/tmp/gocache-goai GOTOOLCHAIN=local CGO_ENABLED=0. Full go test ./backend/cpu default and GOEXPERIMENT=simd; focused GELU/Activation tests GMP1/12 bothmodes; go build ./... bothmodes; go vet ./backend/cpu bothmodes; GOARCH=amd64 GOEXPERIMENT=simd go test -c ./backend/cpu to NEW artifact path; CGO_ENABLED=1 focused race bothmodes; gofmt and git diff --check. Logs in parent artifact dir with v2- prefix. Inspect emitted ARM64 helper instructions for real early branch BEFORE exp/P/Q, extract overhead, mixed continuation, new spills/calls/code size; no new scalar math.Erf/Exp helper in eligible vector body. Root commits restored candidate; then build NEW /private/tmp/goai-m2-f64-gelu-neon-26BhJN/gelu-v2.test, recording source/binary/harness/runtime hashes. Do NOT overwrite immutable V1 binary. Stop CPU-heavy work for independent verifier and campaigns. Fresh verifier independently reruns VERIFY from diff, never trusts implementer transcript.

PERFORMANCE root-only: unchanged run.sh and old control, new V2 binary. Complete all3 order-balanced count7 campaigns and fixed controls with no concurrent CPU-heavy work. Large n262144 public forward/backward active/mixed eachcampaign >=1.25x serial and>=1.05x parallel p<.05; no repeatable>3%small/fixed-control regression or allocationincrease. Retain all samples, errors, ambient-load disclosures, p-values, hashes and independent verdict in separate V2 evidence paths. Reject/revert unproven runtime; no subset/geomean/threshold relaxation. Parent archives only after final verification and all CI jobs AND steps including softSIMD lanes pass, then authorized merge and exact remote-branch deletion. Report generalizable findings to existing perfscan966.
