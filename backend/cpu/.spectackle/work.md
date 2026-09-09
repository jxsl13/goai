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

## T-01M23QYMYVE0MSHXZMCARBFDN2 Capture exact fixed-count control allocation totals with identical-binary negative controls
kind: task
state: draft
created: 2026-09-09
parent: P-01M23BZPZNF9VV3FF326AQX28E
refs: R-01M23Q8QJDE3Y878N2CZBNPAWJ
grilled: 2026-09-09 open=0
targets: backend/cpu/control_alloc_diagnostic_test.go

Diagnostic-only follow-up consuming R-01M23Q8QJDE3Y (retained allocation-research.txt at050ebf55). Rejected V1/V2 and ARM64-F64-GELU-CONTROLS-001 stay unchanged. No runtime promotion/rescoring, selective reruns, threshold changes, CPU scheduler tuning or new allocation optimization. Purpose: determine whether process-wide rounded B/op movement appears with identical binaries and whether matched equal-N old/V2 raw totals differ.

OWNERSHIP: fresh implementer owns ONLY new backend/cpu/control_alloc_diagnostic_test.go in /private/tmp/goai-m2-f64-gelu-neon-26BhJN/repo. Root owns runner, analysis, evidence/docs, lifecycle, commits/pushes, builds of matched pinned old/V2 diagnostic binaries and all campaigns. Do not edit existing benchmark functions/harness, runtime, frozen binaries or .spectackle bytes. Read .claude/commands/spectackle.md fully and get task. apply_patch edits. No commits/lifecycle/mutations or heavy work without root coordination. Assigned isolated branch; don't create another worktree.

IMPLEMENT: package cpu_test, untagged new test-only file. An explicit GOAI_ALLOC_DIAGNOSTIC=1 opt-in guard must skip ordinary CI/fulltest runs BEFORE any Benchmark invocation. TestCPUControlAllocationDiagnostics requires test.benchtime exactly1024x and runtime.GOMAXPROCS(0)1or12 (validate before measuring), then sequentially invokes testing.Benchmark on the exact unchanged existing BenchmarkSiLUBackwardF64_256K_cpu, BenchmarkSigmoidF64_64K_cpu, BenchmarkSoftplusF64_256K_cpu in that order. These existing funcs call benchOn/benchActBwd and original production Execute; do not duplicate loops or memory snapshots. Capture BenchmarkResult.N,T,MemBytes,MemAllocs. testing.Benchmark's private output writer discards error output, so wrap each call with deferred b.Failed detection (atomic.Bool or rigorously synchronized result) and propagate any failure to t.Fatal; nested b.Fatal/Goexit must be caught via defer. No measured-loop code/extra metrics/force-GC changes. Print one machine-readable JSON record per completed control AFTER Benchmark returns, prefixed allocdiag: . Fields exactly benchmark,procs,n,total_ns,total_bytes,total_allocs,bytes_per_op,bytes_remainder,allocs_per_op,allocs_remainder. Validate N=1024,T>0 and no B/op or allocs/op Extra overrides; totals are uint64 not float, use exact quotient/remainder. No JSON/test output inside timed region. The run1 one-iteration calibration is excluded by testing; disclose it. No direct start/end MemStats requirement; public result totals are exact existing timing-bracket deltas. Results remain process-wide, not site attribution.

UNIT TESTS: verify guard pure configuration helper for disabled,valid,wrongN/GMP values; exact formatting/arithmetic helper using synthetic totals with nonzero remainders and values>2^53; no overrides; invalidN/time; ensure no benchmarks run by default. Include a bounded nested-failure unit proof if needed by helper injection or a root-approved explicit guarded diagnostic command; do not run expensive controls in ordinary tests. Prefer helpers returning errors so test can assert behavior without mutating test flags. Do not invoke nested tests in parallel; flag values read-only.

VERIFY implementer: pinned Go1.27.1 PATH/GOCACHE=/private/tmp/gocache-goai/GOTOOLCHAIN=local/CGO_ENABLED=0. Focused new unit/disabled guard tests default and GOEXPERIMENT=simd; full CPUtests bothmodes; CPUvetbothmodes; gofmt-d and diffcheck. Capture raw logs under /private/tmp/goai-m2-f64-gelu-neon-26BhJN/allocdiag-implement-*.log. DO NOT run opt-in benchmark diagnostic or profile/build frozen artifacts; root/fresh verifier coordinate those after review. Fresh verifier independently reruns from diff and tests nested failure propagation with bounded synthetic benchmark, confirms old/new diagnostic source exactlysame and no production modifications.

ROOT preregistered measurement: original old source dd1e779eb085bb621ed5dafff0a4351636b6e656 and V2runtime361955ac35eaada696045f2cf828817b650174ec plus exact same added diagnostic file, Go1.27.1-X:simd DarwinARM64/v8.0 CGO0. Separate new matched diagnostic binaries; old/V1/V2 original binaries immutable. PhaseA identical old diagnostic binary at both A/B labels; phaseB old vsV2 matched diagnostic binaries. Freeze all hashes before running. Eachphase3campaigns×7pairs×GMP1/12×2arms; GMPorder1,12 inodd campaigns,12,1 ineven; armorder A,B if(pair+campaign)even else B,A. All3controls per invocation; exact1024x allcells. Eachphase84invocations/252records; retain every JSON+PASS/FAIL. No extra warmup beyond testing's normal run1 calibration; explicit cold-process, calibrated-loop accounting diagnostic, not warm/cold latency leadership. Separate serialized phases with no ownedbuild/test/profile/benchmark overlap. Total504records. Same named original funcs and no selection by outcome; phaseB runs regardless of phaseA finding if protocol valid. Invalidexecution retained and investigated, never silently replaced.

ROOT ANALYSIS preregistered: reject missing/duplicate/order/PASS/hash/N/JSON/arithmetic mismatch; analyze both phases per campaign/procs/control, exact paired total_bytes/total_allocs deltas B-A and all7rawpairs, medians/ranges, positive/zero/negative counts, rounded B/op and remainders; exact tied-rank p descriptiveonly. No subtraction/noise-floor adjustment, equivalenceclaim, retroactive V2promotion or statistical-significance-as-equality. Old-old repeatedmovement demonstrates possibility without code delta; absence cannot prove noiseabsent. Candidate-specific totaldelta requires allocation-site followup before causal attribution. Any future gate revision needs separate reviewed proposal and new prospective completequalification. Fresh independent report verifies bothphases and unchanged V2verdict. Rawdata, environment,limitations,hypotheses are retained and generalizable lessons reported to perfscan.
