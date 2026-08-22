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

## R-01M0N4TXJQE4NTD97E585SHFTP Measure production-shape M2 Q6_K roofline and geometry leverage
kind: research
state: active
created: 2026-08-22
targets: backend/metal/q4k_roofline_bench_test.go, msl:qmatmul_q6k_cooperative, objc:metal_bridge.mtl_recorder_qmatmul

Q6_K accounts for about 16 percent of profiled TinyLlama decode GPU time while the current cooperative kernel uses two output rows per SIMD group and two SIMD groups per threadgroup. Establish GPUStartTime-to-GPUEndTime throughput at the production 2048x2048, 2048x5632, and 5632x2048 shapes plus a cache-busting shape. Compare current weight-byte bandwidth against the retained Q4_K roofline before nominating a distinct rows-per-SIMD or threadgroup-geometry experiment. This does not repeat the rejected aligned ushort packed-load candidate.

## P-01M0N4ZCJ1FRNT7FDZ03P1BN4W Increase M2 Q6_K memory-level parallelism with rows-per-SIMD specialization
kind: proposal
state: active
created: 2026-08-22
refs: R-01M0N4TXJQE4NTD97E585SHFTP
targets: msl:qmatmul_q6k_cooperative, objc:metal_bridge.ensure_qmatmul_q6k, objc:metal_bridge.mtl_recorder_qmatmul, backend/metal/q6k_roofline_bench_test.go

GPU-timestamp roofline measurements on Apple M2 Pro put the current Q6_K cooperative kernel at 191.3 GB/s for K2048 N2048, 214.6 GB/s for K2048 N5632, 208.0 GB/s for K5632 N2048, but only 157.0 GB/s for a cache-busting K2048 N131072 stream. The retained Q4_K cache-busting result is about 185 GB/s. The Q6_K kernel presently assigns two output rows to each SIMD group, retaining two accumulators and serially consuming two independent weight rows while reusing a small activation tile. Specialize otherwise identical one-, two-, and four-row variants and compare them in one binary. The candidate changes output-row parallelism and register pressure only; it must not repeat the rejected aligned ushort packed-load transform. Promote only a geometry that preserves exact route scope and 2e-5 numerical parity and reaches at least 1.10x control in every representative production cell across three count-seven alternating campaigns, with a cache-busting roofline improvement corroborating the mechanism.

## T-01M0N51GZ1FC1B3VQETC8FDZMN Implement and gate Q6_K rows-per-SIMD variants
kind: task
state: active
created: 2026-08-22
parent: P-01M0N4ZCJ1FRNT7FDZ03P1BN4W
refs: R-01M0N4TXJQE4NTD97E585SHFTP
targets: backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/metal.go, backend/metal/q6k_roofline_bench_test.go, backend/metal/q6k_cooperative_test.go

Refactor the existing Metal Q6_K cooperative body into compile-time one-, two-, and four-output-row specializations without changing byte decoding, arithmetic order within a row, M>1 fallback, or the rejected packed-load strategy. Add an unexported same-binary selector and a floor proving distinct route arms. Validate the specialized outputs against the retained two-row control and the GGUF reference through K=256 to 5632, including finite and nonfinite classification and input immutability. Run three independent count-seven alternating M2 campaigns over K2048 N2048, K2048 N5632, and K5632 N2048 using GPU command timestamps and recorder wall time; retain a new default only if every production cell is at least 1.10x control. Confirm the cache-busting K2048 N131072 roofline moves in the predicted direction and measure full TinyLlama decode before promotion.
