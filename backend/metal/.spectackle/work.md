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

## P-01M0H0SG6FFVKBE666TCKH10MG Pack Q4_0 quant bytes into aligned ushort loads
kind: proposal
state: active
created: 2026-08-21
targets: msl:qmatmul_q4_0_cooperative, objc:metal_bridge.ensure_qmatmul_q4_0, objc:metal_bridge.mtl_qmatmul_resident, objc:metal_bridge.mtl_recorder_qmatmul

On Apple M2, test an alignment-safe Q4_0 cooperative decode kernel that replaces 32 scalar uchar reads per 32-weight block with eight aligned ushort reads and SIMD shuffles. The proof relies on the buffer base, 18-byte row/block strides, and 2-byte q offset preserving two-byte alignment. Retain the candidate only behind an explicit M=1 shape gate when three independent same-binary count-7 campaigns show at least 1.10x median speedup for every eligible production shape, numerical parity stays within 2e-5 relative error, nonfinite classes match, inputs remain immutable, and all direct, resident, Recorder, fallback, and platform tests pass. Otherwise reject and revert the product diff.

## T-01M0H0T9FNEWPBMJKESF613K7D Implement and gate Q4_0 SIMD pair loads
kind: task
state: active
created: 2026-08-21
parent: P-01M0H0SG6FFVKBE666TCKH10MG
targets: msl:qmatmul_q4_0_cooperative, objc:metal_bridge.ensure_qmatmul_q4_0, objc:metal_bridge.mtl_qmatmul_q4_0, objc:metal_bridge.mtl_qmatmul_resident, objc:metal_bridge.mtl_recorder_qmatmul

Add a control-preserving Q4_0 cooperative candidate that loads the 16-byte quant plane through exactly eight aligned ushort device reads and distributes each packed pair with SIMD shuffle. Expose one shared test toggle and route predicate across direct, resident, and Recorder paths. Prove alignment, numerical and nonfinite parity, input immutability, eligible and fallback dispatch counts, then run three independent same-binary AB/BA count-7 M2 campaigns. Promote only if every eligible shape is at least 1.10x control and fallbacks stay within three percent; otherwise revert product code and archive the rejection evidence.
