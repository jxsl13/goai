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

## R-01M0QF5B7EE31TSGZKGEMY4YEZ Fuse M2 split-K chunk processing and merge inside one threadgroup
kind: research
state: draft
created: 2026-08-23
targets: backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/metal.go, backend/metal/mha_decode_bench_test.go, llamagpu

The rejected lane-owned pass-2 candidate proved that pass-2 redundancy exists but is too small in isolation: only 2 of 8 full-attention cells cleared the frozen 1.05x median gate. The higher-leverage boundary is the two-dispatch architecture itself. For dk=64 decode, launch one threadgroup per head with nchunk SIMD groups. Each SIMD group preserves the production lane-quad pass-1 arithmetic for one key chunk, writes its 66-float partial to threadgroup memory, and synchronizes. SIMD group 0 then merges chunk partials in ascending chunk order with lane-owned output dimensions and writes O. This removes the global PART write/read round trip, the second command encoder, and the process-global scratch dependency while retaining the current split-K parallelism within each head. Support the default nchunk <= 16 and fall back when pipeline thread limits reject nchunk*32 threads. Validate both f32 and f16-KV production lanes with same-command GPU timestamps and paired TinyLlama decode.

## P-01M0QF5QXRF21BMTQ7ZF9BAB3B Collapse M2 split-K decode attention into one fused threadgroup kernel
kind: proposal
state: active
created: 2026-08-23
grilled: 2026-08-23 open=0
targets: backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/metal.go, backend/metal/mha_decode_bench_test.go, llamagpu

Replace the default lane-quad two-pass split-K implementation for eligible dk=64 decode with opt-in f32 and f16-KV fused kernels. One threadgroup is assigned per head and contains one 32-lane SIMD group per key chunk. Each SIMD group executes the incumbent lane-quad chunk arithmetic, stores m/l/64 accumulators into dynamically sized threadgroup memory, and synchronizes. SIMD group 0 merges chunks in incumbent order with dimension-owned lanes and writes the output. The candidate removes global PART traffic, process-global scratch allocation, and the second encoder while preserving a correct two-pass fallback when the toggle is off, shapes are out of scope, or nchunk*32 exceeds the pipeline limit. Promotion requires deterministic numeric parity, distinct profile routing, three same-command count-7 M2 campaigns over f32/f16-KV sk 512/1024/1536/2048, and three valid paired TinyLlama campaigns at context 512/1536. This consumes R-01M0QF5B7EE31 and supersedes the rejected isolated pass-2 proposal P-01M0QE1VX8FKN.
