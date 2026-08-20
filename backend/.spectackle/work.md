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

## ADR-01M0FS4JSXFD4TBKY3ESMRGSQ8 How should synchronous host-resident Vulkan bias-gradient reduction choose its execution side?
kind: adr
state: done
created: 2026-08-20
context: The incumbent Vulkan route predates the later bit-exact parallel CPU kernel. The current API returns host tensors synchronously, while future recorder/device-buffer graphs have a different residency contract.
decision: Use a measured shape crossover and preserve Vulkan outside proven CPU winner zones
consequences: Production routing changes only where three independent M2 campaigns clear the frozen speed and stability gates. The old Vulkan path remains the control and the fallback outside measured CPU zones. This applies only to synchronous host-resident tensors; recorder/device-buffer execution remains Vulkan-resident.
status: accepted

kind: radio
option: Always use Vulkan to preserve nominal backend affinity
option: Always route F32 reductions to CPU
option: Use a measured shape crossover and preserve Vulkan outside proven CPU winner zones
blocks: T-01M0FS3HRCE44AKTADYZEQVADR
choice: Use a measured shape crossover and preserve Vulkan outside proven CPU winner zones

## P-01M0G1P2CTFSRS8R2KZ73M78PW Complete M2 synchronous Metal unary routing after CPU specialization
kind: proposal
state: active
created: 2026-08-20
grilled: 2026-08-20 open=0
targets: go:metal.unaryF32, c:mtl_unary_f32, backend/cpu/elementwise.go, backend/cpu/vexp_arm64.go

Historical T535 correctly rejected broad host routing while unary CPU alternatives used scalar closure kernels. The CPU backend now has devirtualized parallel Neg, ReLU, Sqrt, and Abs kernels plus typed arm64 SIMD Exp, Log, Tanh, and Sigmoid kernels. Revalidate the eight remaining Metal unary operations independently in default and GOEXPERIMENT=simd builds across decode-sized, training-sized, and large contiguous F32 tensors. Require three isolated count-7 production-selector campaigns with at least 1.10x median speedup in every routed cell, exact or operation-specific numerical parity, mutation tests for every selected and preserved arm, and at least 0.99x throughput in an affected end-to-end workload. Preserve direct Metal for unmeasured architectures, layouts, sizes, build modes, and operations. Report all generalizable routing or harness findings to perfscan.

## T-01M0G1PRH9E13VST4PPCK2QGF4 Benchmark and gate the remaining M2 Metal unary routes
kind: task
state: active
created: 2026-08-20
parent: P-01M0G1P2CTFSRS8R2KZ73M78PW
targets: go:metal.unaryF32, c:mtl_unary_f32, backend/cpu/elementwise.go, backend/cpu/vexp_arm64.go

Add same-binary direct-Metal controls and production-selector candidates for Neg, Exp, Log, Tanh, ReLU, Sigmoid, Sqrt, and Abs. Benchmark default and GOEXPERIMENT=simd builds independently across 2,048 through 4,194,304 contiguous F32 elements, isolating operations by process with 20 untimed warmups and three independent 100x count-7 campaigns. Route only operation/build/size cells whose every median clears 1.10x; retain direct Metal for unmeasured builds, architectures, layouts, and sizes. Prove reference parity using existing exact or SIMD tolerance contracts, pin selected and preserved selector arms, guard autograd single-record behavior for differentiable routes, and measure at least one affected end-to-end workload at a 0.99x floor. Publish evidence and report generalizable findings to perfscan.

## ADR-01M0G1PTQPF5795JMHPQFY6Y50 Choose execution sides for the remaining synchronous Metal unary operations
kind: adr
state: done
created: 2026-08-20
parent: P-01M0G1P2CTFSRS8R2KZ73M78PW
decision: Use operation- and build-specific measured CPU ceilings with direct Metal outside each frozen winner zone
consequences: Default arm64 routes Neg/Sqrt/Abs through 4,194,304 elements, ReLU through 65,536, and Exp/Log/Tanh/Sigmoid through 2,048. arm64 SIMD extends Exp/Log/Tanh/Sigmoid through 4,194,304. Only valid contiguous offset-zero F32 tensors route; Intel Darwin, invalid or empty inputs, views, larger sizes, and unlisted operations preserve direct Metal. Any kernel or wrapper implementation-class change invalidates only the affected matrix cells and requires same-binary remeasurement.
status: accepted
targets: go:metal.unaryF32, c:mtl_unary_f32, backend/cpu/elementwise.go, backend/cpu/vexp_arm64.go

Decide independently for Neg, Exp, Log, Tanh, ReLU, Sigmoid, Sqrt, and Abs and independently for default versus arm64 SIMD builds among direct Metal, optimized CPU, or a measured size-bounded selector. Historical T535 remains binding until same-binary production-selector campaigns, numerical parity, selector mutation tests, recorder safety, and an affected workload floor establish a new winner zone.
choice: Use operation- and build-specific measured CPU ceilings with direct Metal outside each frozen winner zone
