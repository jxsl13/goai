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

## P-01M0FQ0RWQE8PT2GT7FFP4WKBE Route host-resident Vulkan embedding backward deterministically
kind: proposal
state: active
created: 2026-08-20
grilled: 2026-08-20 open=0
targets: backend/vulkan/vulkan.go, backend/vulkan/embed_backward_route_bench_test.go

Context: merged Metal evidence P-01M0FNKC7DEJZ proved that host-resident synchronous embedding backward loses to a typed host scatter. Vulkan currently repeats the same upload of indices, upstream gradient, and zero table, atomic dispatch, synchronization, and full-table download. ADR-01M0FQ1AP8FBC conditionally selects the typed host route while persistent device residency remains a separate graph redesign. Base 0911d9d33fe65fa6023d83efe042556f36a00bc1. Frozen M2 Pro gate: preserve the current Vulkan atomic path as a same-binary control; exact reference parity including repeated indices; benchmark n513/d128/m520, n65/d128/m520, n520/d128/m8, n4096/d512/m128, and n32768/d128/m512 in three independent seven-sample campaigns through MoltenVK; every median must improve by at least 1.20x and candidate spread stay at most 3.0x; no API or portable semantic change; full Vulkan and repository suites green. Retain only a measured winner and update perfscan#771 with sibling evidence.

## ADR-01M0FQ1AP8FBCSCA67X0RCPS8A Where should Vulkan OpEmbedBackward execute while the backend contract returns host-resident tensors synchronously?
kind: adr
state: done
created: 2026-08-20
context: The sibling Metal route has already passed exactness and 3.93x to 30.76x M2 gates; Vulkan must still pass its own MoltenVK measurements before sharing the decision.
decision: Typed deterministic host scatter at the current boundary
consequences: Vulkan OpEmbedBackward will use zero Vulkan submissions only if its independent MoltenVK campaigns clear every frozen speed and spread gate. Reference-order F32 accumulation becomes deterministic. Persistent device residency remains a separate graph-level redesign and invalidates this route decision when introduced.
status: accepted

kind: radio
option: Typed deterministic host scatter at the current boundary
option: Vulkan atomic scatter with upload, wait, and full-table download
option: Introduce persistent device-resident embedding state in this slice
blocks: P-01M0FQ0RWQE8PT2GT7FFP4WKBE
choice: Typed deterministic host scatter at the current boundary

## T-01M0FQ2Z5MFQ1V30BDS7GGRDTF Gate deterministic host Vulkan embedding backward
kind: task
state: active
created: 2026-08-20
parent: P-01M0FQ0RWQE8PT2GT7FFP4WKBE
targets: backend/vulkan/vulkan.go, backend/vulkan/embed_backward_route_bench_test.go

Implement a typed F32 host scatter in Vulkan embedBackwardF32, preserve reference per-add rounding and repeated-index accumulation, preserve the old SPIR-V atomic path as a same-binary control, run the five frozen M2 MoltenVK shapes across three independent campaigns, validate the full Vulkan and repository suites, update perfscan#771, and use a dedicated PR merged only after every CI check passes.
