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

## R-01M0Q913MDFQSTD3ME8MTRCGRZ Recorder profile label ownership and allocation scaling
kind: research
state: active
created: 2026-08-23
targets: go:metal.Recorder.Profile, backend/metal/recorder_profile_bench_test.go

The merged bulk snapshot leaves 344 allocations and 19,816 B per warm 340-event extraction on M2 Pro. Code inspection attributes 340 allocations to C.GoString, one per event, although repeated kernels share labels. Investigate recorder-scoped owned label reuse: compare native labels without retaining C memory, clone each distinct label once, reuse the Go-owned value across Profile calls, preserve returned labels after Recorder.Free, and leave default recorders allocation-neutral. Freeze repeated-label, one-event, first-extraction, mixed-label, ownership, and disabled-recorder gates before implementation.

## P-01M0Q9EZJ8ENCSJ4YR1J41MDGD Deduplicate labels within each Recorder.Profile extraction
kind: proposal
state: active
created: 2026-08-23
refs: R-01M0Q913MDFQSTD3ME8MTRCGRZ
grilled: 2026-08-23 open=0
targets: go:metal.Recorder.Profile, go:metal.Recorder.Free, backend/metal/recorder_profile_bench_test.go

Supersedes rejected P-01M0Q9837XE0Y. Replace one C.GoString allocation per event with extraction-scoped Go-owned label deduplication. For multi-event profiles, use a temporary bounded view of each 96-byte native label only for lookup and clone each distinct label once into the returned profile; results remain valid across Recorder.Free. Keep the existing C.GoString one-event path unchanged and add zero fields or allocations to default recorders. Frozen M2 gates: warm repeated-label events340 median speedup at least 1.25x with at least 300 fewer objects and 4,000 fewer bytes; every warm events1 campaign retains at least 0.97x; first events340 at least 1.10x and first events1 at least 0.97x; mixed-label allocations scale with distinct labels rather than event count; disabled-recorder throughput at least 0.97x with unchanged allocations. Require three order-alternated count-seven campaigns, exact profile parity, bounded-label handling, and label validity after Free plus GC churn.

## T-01M0Q9H5G2FP38JV8VMPSFP297 Implement and gate extraction-scoped profile label deduplication
kind: task
state: draft
created: 2026-08-23
parent: P-01M0Q9EZJ8ENCSJ4YR1J41MDGD
refs: R-01M0Q913MDFQSTD3ME8MTRCGRZ
targets: go:metal.Recorder.Profile, backend/metal/recorder_profile_bench_test.go, backend/metal/recorder_profile_test.go

First add a mixed-ten-label benchmark and post-Free ownership test, then compile the frozen control binary. For multi-event profiles, scan each fixed native label within 96 bytes, use temporary unsafe string views only for content lookup, and clone every distinct label once into returned Go memory. Keep count-one C.GoString unchanged. Run parity and ownership tests plus three order-alternated count-seven M2 campaigns for warm repeated/mixed labels, fixed-count first extraction, and disabled recorder overhead. Promote only if every frozen gate passes; otherwise revert code and archive the rejection.
