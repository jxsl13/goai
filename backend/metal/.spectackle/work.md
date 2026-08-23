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

## P-01M0Q6T9RZEV9RWA9ZS5V8KGYC Bulk-extract Metal recorder profile events in one cgo call
kind: proposal
state: active
created: 2026-08-23
grilled: 2026-08-23 open=0
targets: go:metal.Recorder.Profile, c:mtl_recorder_profile_event, backend/metal/metal.go, backend/metal/metal_bridge.m, backend/metal/metal_bridge.h

Research R-01M0Q6SJ8CF6R and rejected scratch proposal P-01M0Q6B7A3E1S show that 340 synchronous cgo event crossings are the remaining leverage. Add an immutable recorder-lifetime native snapshot containing fixed 96-byte labels and three uint64 timing fields per valid event. A new additive C ABI shall resolve and return summary fields plus the snapshot pointer in one cgo call; existing summary/event entry points remain unchanged. Go shall own its returned RecorderProfile and strings exactly as before. Benchmark warm repeat and first extraction for 1 and 340 events. Across three order-alternated count-seven M2 campaigns, warm events340 must be at least 1.50x with at least 1000 fewer allocations and 40000 fewer Go allocation bytes per operation; warm events1 must be at least 1.10x. First-extraction events340 must be at least 1.10x and events1 at least 0.97x. Reject and fully revert if any gate or exact parity fails.

## T-01M0Q6VRADE45AJWQRNNN3JZNC Implement and gate bulk Metal profile snapshots
kind: task
state: active
created: 2026-08-23
parent: P-01M0Q6T9RZEV9RWA9ZS5V8KGYC
targets: backend/metal/metal.go, backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/recorder_profile_bench_test.go, backend/metal/recorder_profile_test.go

Extend the physical-Metal benchmark with first-extraction events1/events340 cells and compile the control binary before implementation. Add a recorder-owned immutable C event snapshot and one additive bulk snapshot function; switch Recorder.Profile to one cgo call and copy into Go-owned values. Verify exact existing profile tests, repeat equality, pre-Finish/default/freed error behavior, overflow and MPS omissions. Run three order-alternated count-seven M2 campaigns for frozen warm and first-extraction gates. Run full tests, direct perfscan, and revert all implementation changes if any gate fails.
