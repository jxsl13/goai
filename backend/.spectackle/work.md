---
schema: v1
---

## T-01KYJREHNVE9QS89QB05TW5SWV Memoize kernel resolution in Execute and stop Metal re-entering the dispatcher
kind: task
state: draft
created: 2026-07-27

DO NOT LAND THIS BEFORE THE ref reduce AND broadcast TASKS. The researching agent measured its own exposure and was explicit: at prefill shapes this is noise — BenchmarkGPTForward/metal is 29 ms for about 93 ops, so roughly 200 ns/op is 0.06%. It only pays at decode shapes. Sequencing matters more than the mechanism here.

WHY IT NONETHELESS MATTERS: on this host backend.Default() is Metal, which serves 39 of 158 ops, so ABOUT 75% OF OPS TAKE THE execute.go:70-104 FALLBACK BRANCH ON EVERY CALL. BenchmarkGemmaDecode issues 35 Execute calls per token (980 per 28-token iteration, counted with GOAI_TIME_OPS=1) and averages 406 ns per Execute at dim=16, where roughly 40 ns of fixed cost is about 10%.

DEFECT, all confirmed by go build -gcflags=-S ./backend/: execute.go:62 calls runtime.mapaccess2_fast64 for the always-nil ctx.opBackends. execute.go:78 calls backend.Get, which disassembles to sync.runtime_SemacquireRWMutexR plus runtime.mapaccess2_faststr plus runtime.deferreturn — a shared-cacheline RWMutex acquisition per fallback op, with the defer not fully elided. The same (op, dtype) is looked up THREE TIMES: :70 miss, :79 probe, :102 fetch. execute.go:103 ctx.WithBackend(fb) emits runtime.newobject plus three gcWriteBarrier1 — a heap Context per fallen-back op. Each Kernel() lookup in ref and cpu emits the GENERIC runtime.mapaccess2 rather than _fast64/_faststr, because kernelKey{Op int; Dtype uint8} is a padded 16-byte struct; measured on this host via go test runtime -bench BenchmarkMapAccessHit, a [16]byte key costs 7.03-11.04 ns against 3.75-8.59 ns for int64 and about 0.5 ns for a dense array index. metal.Kernel returns freshly built closures — -gcflags=-m confirms metal.go:523:24 func literal escapes to heap for roughly 20 of Metal's 39 ops, i.e. an allocation on every kernel lookup. Worst: metal.go:660-666 binaryF32 calls cpuPrefers (itself another Get plus another Kernel), then ctx.WithBackend(cpu).WithRecorder(nil) (two more Context allocations), then RE-ENTERS backend.Execute — a complete second dispatch. One OpAdd on a Metal context therefore costs about 6 map lookups, 1 RWMutex round trip, 3 heap allocations, 2 checkAttrs calls and 2 Execute frames.

FIX, four independent sub-changes: (1) memoize resolution in a lazily built, atomically published [numOps][numDtypes]struct{k Kernel; b Backend} per registered active backend, invalidated by Register/RegisterDefault/SetPreference (registration is init-only in practice, so invalidation is a cold path); one array index replaces Get plus three Kernel calls plus the WithBackend allocation. (2) replace map[kernelKey]Kernel in ref/ref.go:47 and cpu/cpu.go:39 with a flat []Kernel of length numOps*numDtypes plus a range guard. (3) hoist Metal's closure factories to package-level vars and return those. (4) have metal.Kernel DECLINE OpAdd/Sub/Mul/Div/Maximum/Minimum when cpuPrefers(op, dtype) holds, so Execute's own fallback chain lands on cpu in ONE dispatch — exactly equivalent, since cpuPrefers is already checked first inside binaryF32 before any shape or dtype test.

CONSTRAINT PRESERVATION, explicitly: checkAttrs (execute.go:40-44) stays where it is, still keyed off opAttrsSpec and still rejecting a wrong concrete type BEFORE resolution — the memo table is consulted after it, and the sealed Attrs interface is untouched. Op remains a typed enum, never a string key. The recorder strip at execute.go:114-117 is unchanged, and removing Metal's inner Execute re-dispatch makes the double-record hazard STRUCTURALLY impossible for those ops rather than relying on the defensive WithRecorder(nil).

VALIDATION GATE (benchmark only): NOTHING EXISTING ISOLATES THIS, and the one benchmark that would is broken — see the separate task for BenchmarkGPTDecode. Fix that first; decode (seq=1) is the only regime where per-op fixed cost is visible. Then write BenchmarkExecuteDispatch in backend/, running OpAdd on a Shape{1,1} F32 tensor through Execute on a ref context, a cpu context and a metal context — the near-zero kernel makes the residual the dispatch cost itself. Land the four sub-fixes separately.

EXPECTED, per op: about 10 ns (map to array), 25 ns (fallback memoization including one 48 B allocation), 15 ns (Metal closure allocation), 150 ns (Metal binary double-dispatch). High confidence on the mechanism, LOW-TO-MEDIUM on end-to-end impact until a working decode benchmark exists.

BIT-IDENTITY BAR: ZERO NUMERIC RISK — no arithmetic is touched anywhere. The risks are semantic and each is testable: the memo must reproduce the exact cpu-then-reference preference order of execute.go:77-94 including the cpu != ctx.Backend guard (backend/routing_test.go and preference_test.go cover this); invalidation must be correct if any program calls SetPreference after contexts exist (the doc at registry.go:112 says call it once at startup — assert that); and declining ops in metal.Kernel changes what Available()-style introspection reports for Metal, so confirm backend/metal/zzz_fallbackaudit_internal_test.go still means what it claims.

## T-01KYMYXXRNEK7AWD5WCKTKCN9T Re-apply the reduceKernel strip-mine for a KEPT innermost axis, on top of main's isSum+trailing path
kind: task
state: draft
created: 2026-07-28

DELIBERATELY REVERTED during the origin/main merge at 92f76c13, and this task is the record of what was given up. This branch had strip-mined backend/ref/reduce.go's odometer; main had landed two orthogonal optimizations the branch lacked. Taking the branch's version would have silently reverted the trunk, so main's was kept whole.

WHAT MAIN HAS: (1) an isSum flag threaded through reduceKernel that inlines `acc += x` rather than paying an indirect call through the combine func value on every element — Go cannot inline through a func value; (2) a trailing-contiguous fast path, taken when the reduced axes are exactly the innermost suffix, that accumulates each output segment in a register.

WHAT IS STILL ON THE TABLE: main's trailing path subsumes the reverted work's reduce-all case and its stride-0 case. The remaining gap is the KEPT-innermost-axis case — reduced axes that are NOT the innermost suffix, e.g. reducing axis 0 of [rows,cols]. That shape falls to main's per-element odometer, which ticks a full O(nd) carry chain per element. The reverted code handled it by hoisting the innermost axis out of the odometer and walking consecutive accumulators (`dst[j] = combine(dst[j], v)` over a re-sliced run), so the odometer ticks once per run.

WHY IT WAS NOT COMPOSED AT MERGE TIME: reduce is a numerics hot path whose bit-identity argument depends on each accumulator seeing the same values in the same ascending order. Splicing two independently-developed traversals together and asserting that property without a benchmark or a fresh gate would have been a guess. Merges are the wrong place to introduce unmeasured optimizations.

HOW: keep main's structure exactly — the trailing branch and both isSum arms — and replace only the `else` per-element odometer with a strip-mined form that hoists shape[nd-1]. Preserve the isSum split inside it, since that is main's win and it is orthogonal. Handle eff[nd-1] == 0 (run folds into one accumulator) and == 1 (run walks consecutive accumulators); decline anything else to the existing odometer rather than carrying an unreachable third branch.

BIT-IDENTITY: unchanged by construction if done correctly — every accumulator must see the same values in the same ascending order, only the index bookkeeping moves. Do not take that on faith. Probe it: perturb one accumulation by a single ulp and confirm the backend/ref reduce tests turn red BEFORE trusting them, per the pattern that found five blind kernels in this package.

MEASURE: a reduction over axis 0 of a 2-D tensor is the shape that exercises the gap; the existing BenchmarkSumF64_64K does NOT reach it (verified by panic probe when the reverted version was written). A new benchmark is part of this task, not optional. Interleave per PROC-INTERLEAVE-001 and discard unless each arm holds near 5%.

RECOVER THE PRIOR CODE from git: the reverted implementation is in this branch's history before the merge commit 92f76c13, in backend/ref/reduce.go. It is a starting point, not a patch to apply — main's file has changed shape around it.

SCOPE: backend/ref only. NOTE the user works in parallel on backend/cuda and backend/cpu amd64 branches; backend/ref has been collision-free so far, but fetch-rebase before starting.

## R-01KYN0Y3EPEV2TAD5AQP5Q74ZG Quantized prefill spent ~69% in scheduler wakeups, not compute — attribution and what is left after the QMatMul fix
kind: research
state: draft
created: 2026-07-28

Recorded because the measurement is in backend/cpu's zone, which this line of work deliberately does not touch, and because the attribution chain took several pprof hops that should not be repeated.

BEFORE the QMatMul weight-row parallelization, a CPU profile of quantized Mamba2 prefill (2 layers, d_model 256, seq 128, Q4_K) read:
  pthread_cond_signal  20.92% flat
  gguf.QMatMul         19.61% flat
  pthread_cond_wait    12.42%
  runtime.usleep       12.42%
  runtime.kevent       11.76%
  runtime.madvise      11.76%
Roughly 69% in runtime synchronization and memory management against 20% of actual matmul.

ATTRIBUTION CHAIN, via pprof -peek: pthread_cond_signal <- semawakeup <- notewakeup <- startm <- wakep <- ready <- runtime.send.goready.func1. That last frame is a CHANNEL SEND waking a parked receiver — the signature of a channel-dispatch worker pool, not of GC or of the profiled code itself.

NOT THIS LINE OF WORK'S POOL, tested rather than assumed: disabling internal/parallel entirely in QMatMul changed prefill by nothing (22.4/23.0/22.3ms off vs 22.4/22.4/22.1ms on). Prefill took the m>1 general path, which did not use that pool at all. The remaining candidate is backend/cpu's parallelWork dispatch, whose own comments already record that naive parking cost a full M-stop per barrier.

CORROBORATION: total CPU samples (1.53s) matched wall time (1.52s) on a 12-core host, and prefill did not scale across GOMAXPROCS 1..12 (22.1 / 21.3 / 21.4 / 22.6ms). Work that parallelizes does not look like that.

AFTER parallelizing QMatMul's weight-row loop, prefill is 2.52x faster and scales 20.8 / 14.0 / 9.8 / 8.5ms at 1/2/4/12 Ps. The scaling curve flattens past 4 Ps, which is where the remaining serial fraction AND this synchronization cost now live. That flattening is the measurable residue of the effect above.

WHY NO ACTION TAKEN: backend/cpu is the zone a parallel worker is active in, and its pool carries heavily tuned dense/sparse regime machinery whose constants were calibrated against decode-shaped barrier streams. A change there needs that context and that worker's benchmarks, not a drive-by from this side. The actionable question for whoever picks it up is whether the ops a QUANTIZED prefill issues (small per-layer norms and elementwise work between large matmuls) fall below poolDenseMaxWork and thrash the regime detector — this workload did not exist when those constants were set.

Recorded as measurement plus attribution, no proposed fix.
