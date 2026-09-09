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

## T-01M24137G7EHMASPJH0JA8ZZKS Implement guarded raw Softplus allocation-site capture with exact artifact schema
kind: task
state: draft
created: 2026-09-09
parent: P-01M23BZPZNF9VV3FF326AQX28E
refs: R-01M23VWNB5FRP85PW5NSQWY5QA
grilled: 2026-09-09 open=0
targets: backend/cpu/control_alloc_sites_test.go

GOAL: implement matched, opt-in, test-only raw allocation-site capture for the reviewed Softplus diagnostic, without changing runtime code or claiming performance/attribution. Consume R-01M23VWNB5FRP; parent P-01M23BZPZNF9V stays active and both GELU rejections plus ARM64-F64-GELU-CONTROLS-001 remain binding. Architecture remains the existing Go Context/Execute CPU path; no dispatch/kernel/storage/scheduler changes or dependencies.

READ FIRST: .claude/commands/spectackle.md in full, spectackle call -instructions, this T and parent/R, and complete files internal/benchcompare/leadership/evidence/m2-control-alloc-sites-20260909/protocol.md, schema.md, corrected-review.txt. schema.md at commit37574269 fixes all capture fields, flags, counts, resource and serialization boundaries. These documents are required instructions, not optional notes. Report any contradiction before implementation; do not silently improvise. Root owns analyzer/runner specification and later matched builds/campaigns.

EXACT WRITE SCOPE: create only backend/cpu/control_alloc_sites_test.go, package cpu_test, plus a report path supplied by root. Use apply_patch, gofmt permitted. No edits to existing helpers/benchmarks/source, go.mod, prior raw evidence, .spectackle bytes or Git. Scope leases may be claimed/released; lifecycle and Git belong to root. Dedicated worktree path will be supplied on delegation. No subagents or exploration needed.

APIs/source: backend.Get(backend.CPU) returns (backend.Backend,bool); backend.NewContext().WithBackend(be) returns *backend.Context; backend.Execute(ctx *backend.Context, op backend.Op, inputs []*tensor.Tensor, attrs backend.Attrs) returns ([]*tensor.Tensor,error). Reuse bench.RandF64(tensor.Shape{1<<18},3) from github.com/jxsl13/goai/internal/bench. Warm exactly one Execute with OpSoftplus and attrs nil outside accounting. See known backend/cpu/bench_test.go:18-28,54-58 and backend/backend.go:90-96. Do not call testing.Benchmark or duplicate a Softplus implementation. Use standard-library runtime, runtime/pprof, runtime.CallersFrames, runtime.FuncForPC and reflect.ValueOf for the built helper name only during setup. Existing cpu_test registration imports make backend available; explicit blank imports cpu/ref are allowed if needed for standalone identical-file builds. No external modules or unsafe/linkname.

IMPLEMENT: TestCPUControlAllocationSites follows the exact opt-in/config rules and capture schema. Empty selector skips before setup/output; bad nonempty selector/config fails. Parameterize config validation as a pure helper so unit tests do not modify global GC/procs/profiling settings. Validate test.run/count, short flag, GODEBUG/GOGC/GOMEMLIMIT, actual rate/procs, and supplied memprofile/memprofilerate flags. Create a new absolute output directory; reject every existing path. Exclusively open capture.json/tail.json/allocs.pprof before the first boundary; preserve artifacts on every failure and surface write/close errors.

Create //go:noinline allocationSiteExecuteRegion(ctx *backend.Context, ins []*tensor.Tensor) (int,error): its body only loops exactly1024 direct Execute calls with OpSoftplus,nil; increment the returned successful-call count after each success. Return early on Execute error with the actual completed count; no formatting/assertions inside it, and never claim constant1024 after an early error. Discard outputs exactly as the original callback.

Allocate both raw buffers with LENGTH=CAPACITY=65536 and all MemStats destinations before warmup. Fixed capture order: warmup; runtime.GC; ReadMemStats(preRawBefore); MemProfile(pre,true); ReadMemStats(S); region; ReadMemStats(E); runtime.GC; ReadMemStats(postGC); MemProfile(post,true); ReadMemStats(postRaw). No formatting, symbolization, output or assertion inside this sequence. Even a snapshot-capacity failure retains the actual returned count/status and the later boundaries; do not resize or retry. An Execute error is only inspected/formatted after E and post capture. All capture validation/formatting follows raw post capture. Require zero pre-snapshot Mallocs/TotalAlloc movement, 1024 successful calls, unchanged rate1, valid snapshot capacities; record errors and retained raw state before failing.

Serialize exactly schema.md, including every original raw row ordinal/four signed counters/32 hex PCs, all CallersFrames in order, empty arrays instead of null, precise uint64 MemStats counters, exact built caller function name and worker name. No slot-size derivation, aggregate maps, statistical analysis or source-line normalization here: the offline analyzer owns those. Retain malformed raw values rather than repairing them. Guard snapshot slice bounds: use only the returned prefix when n/ok valid; overflow means no copied rows, not zero-filled unused capacity. Write capture JSON, then one cumulative pprof, then ReadMemStats(tail) and tail JSON. Final tail serialization/close costs are explicitly excluded. All artifacts/diagnostic errors survive failure.

TESTS: add focused TestAllocationSite* unit tests in the same file for disabled/no-work selection; bad selectors/rate/procs/debug/GC policy/test flags/short; exact counter JSON roundtrip above2^53 and near uint64 max; negative raw counters retained; all32 PCs retained; order/inline frame lists not sorted/deduplicated; empty-stack arrays; snapshot prefix vs insufficient-buffer handling; output path reuse rejection without overwrite; short-write/write/close failure propagation where helpers expose writers; region early Execute error returns actual count and valid small-input region returns1024. Do not weaken the actual guard or make test-only injected callbacks part of the production region. Test pure formatting/config helpers with synthetic records; do not run MemProfile/pprof or the enabled 256K profile test yet. No timing or allocation-budget assertions in unit tests.

VERIFY serial, pinned Go1.27.1: PATH=/private/tmp/goai-go1271-j1cLvq/go/bin:$PATH GOCACHE=/private/tmp/gocache-goai GOMODCACHE=/private/tmp/goai-go1271-j1cLvq/modcache GOTOOLCHAIN=local CGO_ENABLED=0. Run go version; gofmt new file; git diff --check; go test ./backend/cpu -run '^TestAllocationSite' -count=1 -v default and GOEXPERIMENT=simd; disabled TestCPUControlAllocationSites count1 verbose default/SIMD, requiring SKIP and no artifact directory; full go test ./backend/cpu -short -count=1 default/SIMD; go vet ./backend/cpu default/SIMD; CGO_ENABLED=1 go test -race ./backend/cpu -run '^TestAllocationSite' -short -count=1. Do not enable GOAI_ALLOC_SITE_DIAGNOSTIC or run any profile/benchmark during this task; a separately authorized, retained preflight after independent implementation review will test live capture before matched matrix freeze. This restriction is a deliberate scope boundary, not permission to mark the whole measurement pipeline complete. Retain literal command outputs/exit codes and failures without truncation; final source SHA256 and clear ran/not-ran list. Report completion to root without lifecycle moves or Git; close heavy window explicitly.
