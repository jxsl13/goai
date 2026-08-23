---
schema: v1
---

## T-01KYJREJ34FN9BEZDREAF6WYT8 Fix the broken BenchmarkGPTDecode — it blocks honest sizing of dispatch work
kind: task
state: draft
created: 2026-07-27

BLOCKER, not an optimization. Small, but it gates a whole line of work.

SITE: backend/metal/gpt_test.go:303 BenchmarkGPTDecode.

SYMPTOM: it fails on both sub-benchmarks with: nlp: DecodeStep position 8 does not continue the cache: the next token is at position 9 (cached rows: 9).

WHY IT MATTERS: this is the ONLY benchmark in the tree that measures per-op dispatch cost at decode shapes (seq=1). Prefill benchmarks cannot see it — the researching agent measured BenchmarkGPTForward/metal at 29 ms for roughly 93 ops, so about 200 ns/op of dispatch overhead is 0.06%, i.e. invisible. At decode shapes the same fixed cost is roughly 10% (BenchmarkGemmaDecode averages 406 ns per Execute at dim=16). Without this benchmark the Execute-memoization task cannot be honestly sized, and any claim about it would be an estimate dressed as a measurement.

FIX: diagnose the position/cache-row mismatch. The error says the loop advanced position to 8 while the cache already held 9 rows, so either the benchmark seeds the cache with a prompt and then restarts positions from a stale index, or it reuses one cache across b.N iterations without resetting. Establish which before changing anything — the fix is either resetting per iteration or deriving the position from the cache length rather than from a counter.

VALIDATION: the benchmark runs to completion on both sub-benchmarks and produces a stable tok/s figure across -count=3. Then confirm it actually exercises the intended regime by checking with GOAI_TIME_OPS=1 that per-token Execute counts match the model's layer count, so it is measuring decode and not accidentally re-running prefill.

SCOPE: fix the harness only. Do not optimize anything in the same change — the entire point is to obtain a trustworthy measuring instrument before optimizing, and a benchmark repaired in the same commit as the thing it measures cannot serve as the A/B baseline.

## ADR-01M0FVWNPKEX6917B1N0VBM0FJ How should synchronous host-resident F32 Metal bias gradients route after the CPU reduction optimization?
kind: adr
state: done
created: 2026-08-20
context: Three independent count-7 M2 campaigns show the production CPU selector is 3.263x to 199.71x faster than direct Metal through 2,097,152 elements, with worst candidate spread 1.788x, exact reference parity, and 0.994x end-to-end GPT throughput.
decision: Route measured shapes through CPU and preserve direct Metal above the bound
consequences: F32 rank-2 gradients with positive dimensions and at most 2,097,152 elements use the exact optimized CPU kernel with recorder suppression. Larger, unsupported, or CPU-unavailable cases retain the isolated direct Metal implementation. Future device-resident graph execution requires a new benchmark boundary and does not inherit this host-resident decision.
status: accepted

kind: radio
option: Route measured shapes through CPU and preserve direct Metal above the bound
option: Retain direct Metal for all shapes
option: Remove the direct Metal implementation
blocks: T-01M0FVGM88EWMRQCHFN4B748AV
choice: Route measured shapes through CPU and preserve direct Metal above the bound

## ADR-01M0M9MPAGFM9T4AMA7D97HWJD Which native Metal Q4_1 kernel shape should be the production baseline on M2?
kind: adr
state: done
created: 2026-08-22
context: Q4_1 uses a 20-byte affine block with d and m. Direct decoding preserves compressed residency and avoids transformation traffic. Q4_0 already proves the scalar and cooperative occupancy shapes.
decision: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0
consequences: The kernel decodes Q4_1 directly from 20-byte resident blocks. Synchronous, resident, and recorder paths share the same cached pipelines. Scalar is the capability fallback; cooperative is the M2 path. No dense expansion or transient Q4_0 transformation is allowed.
status: accepted

kind: radio
option: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0
option: Reuse Q4_0 after transforming Q4_1 weights or activations
option: Materialize dense F32 weights before Metal GEMM
blocks: P-01M0M9B6FRFCZA18408PMM2WGH
choice: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0

## R-01M0QCMRDYEBCAT35BDCB8678D Eliminate reusable Metal recorder profile extraction allocations
kind: research
state: draft
created: 2026-08-23
targets: go:metal.Recorder.Profile, go:metal.fillRecorderProfileEvents, c:mtl_recorder_profile_snapshot, backend/metal/recorder_profile_bench_test.go, backend/metal/recorder_profile_test.go

Merged baseline on Apple M2 Pro: warm Profile extraction for 340 repeated-label events costs 2.61-2.84 us, 14,400 B/op, and 6 allocs/op; one event costs about 179 ns, 104 B/op, and 5 allocs/op; 340 events with ten labels costs 3.38-3.52 us, 14,544 B/op, and 15 allocs/op. The dominant bytes are the fresh []RecorderProfileEvent result. Earlier recorder-owned label scratch designs were rejected because they couple result lifetime to Recorder and grow the opt-out path. Investigate an additive caller-owned ProfileInto destination and one compact native snapshot view, preserving Profile ownership, ABI entry points, default-recorder cost, and post-Free label safety. Promotion requires three independent count-seven order-controlled M2 campaigns, exact semantic tests, and existing-path non-regression.

## P-01M0QCN4C8ECWANK89M5CWF4P0 Add caller-reused Metal Recorder.ProfileInto extraction
kind: proposal
state: active
created: 2026-08-23
targets: go:metal.Recorder.Profile, go:metal.fillRecorderProfileEvents, c:mtl_recorder_profile_snapshot, backend/metal/metal_bridge.m, backend/metal/recorder_profile_bench_test.go, backend/metal/recorder_profile_test.go

Consume R-01M0QCMRDYEBC. Add Recorder.ProfileInto(dst *RecorderProfile) error as an additive API. A valid call reuses dst.Events capacity, overwrites all scalar fields, truncates to the native count, and preserves Go-owned labels after Recorder.Free. A nil destination is an explicit error. On native extraction failure, dst remains unchanged. Reuse exact existing destination label strings where the native label content matches, allowing warmed repeated-label calls to avoid label allocation without recorder-owned Go state. Add one native by-value snapshot-view entry point carrying events, label tokens, counts, omissions, calibration, duration, and status; retain existing native entry points for ABI compatibility. Profile remains the convenience ownership API and must retain its one-event and 340-event performance. Promotion gates: three independent order-alternated count-seven Apple M2 campaigns; warmed capacity-sufficient ProfileInto events340 must use at least 10,000 fewer B/op, at least 2 fewer allocs/op, and run at least 1.25x faster than Profile in every campaign; mixed-label warmed reuse must allocate zero new label strings; capacity-insufficient first call may allocate then must stabilize; Profile events1 and events340 must retain at least 0.97x baseline throughput; default recorder allocations remain unchanged; parity, error atomicity, repeated destination reuse, and post-Free ownership must pass.

## T-01M0QCNME5FE0VKFK7MPF9F0G6 Implement and gate caller-owned Metal ProfileInto
kind: task
state: draft
created: 2026-08-23
parent: P-01M0QCN4C8ECWANK89M5CWF4P0
targets: go:metal.Recorder.Profile, go:metal.fillRecorderProfileEvents, c:mtl_recorder_profile_snapshot, backend/metal/metal_bridge.m, backend/metal/recorder_profile_bench_test.go, backend/metal/recorder_profile_test.go

Implement the additive caller-reused API and compact native snapshot view defined by P-01M0QCN4C8ECW. Add parity, error-atomicity, repeated reuse, insufficient-capacity stabilization, and post-Free ownership tests plus warm repeated-label and mixed-label benchmarks. Measure ProfileInto against frozen Profile baseline in three independent order-alternated count-seven Apple M2 campaigns. Retain only if every performance, allocation, ownership, ABI, and non-regression gate passes. Report the generalizable caller-owned output-buffer and cgo-boundary result to perfscan.
