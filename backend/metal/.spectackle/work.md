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

## P-01M0GX86TYEEJSJ2SB873CJDDY Pack aligned Q6_K byte-plane loads on M2
kind: proposal
state: active
created: 2026-08-21
grilled: 2026-08-21 open=1
targets: msl:qmatmul_q6k_cooperative, objc:metal_bridge.mtl_qmatmul_q6k, go:metal.SetQ6KCooperative, go:metal.Recorder.QMatMulResident, backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/metal.go, backend/metal/q6k_bench_test.go

Context: Apple M2 Pro Q6_K M=1 decode is the higher-precision projection leaf used in Q4_K_M models. Fresh post-PR #1116 baselines measured K2048xN256 at roughly 149-210 us, K2048xN2048 at 242-252 us, and K5632xN2048 at 255-341 us. Each cooperative lane currently reads four contiguous bytes from q1, q2, and qh through twelve indexed uchar loads per block. The 210-byte block and row strides preserve only two-byte alignment, so uint, uint2, or ulong casts are invalid for alternating blocks; aligned ushort pairs are safe.

Hypothesis: replace each four-byte uchar group with two aligned ushort loads for q1, q2, and qh, then extract the original bytes in registers while preserving the existing q6 reconstruction and float accumulation order. This is not the rejected Q4_K experiment: Q4_K already used ushort quant loads, whereas Q6_K still issues twelve byte-level bit-plane reads.

Scope: add an independently selectable runtime-compiled cooperative candidate, initially disabled. Preserve scalar and historical cooperative kernels. Use one shared route predicate in direct, resident, and Recorder selectors. Do not alter GGUF layout, numerical semantics, M>1 paths, unsupported-device fallback, or other quant types.

Gates: scalar, control, and candidate outputs must agree within 2e-5 relative error, preserve finite/Inf/NaN classification, and leave inputs immutable. Benchmark identical resident buffers with AB/BA reversal, exclude one transition dispatch, and time at least 32 steady-state dispatches per arm. Cover KV, square, mid-up/down, gate/up, down, and vocabulary shapes. Promote only a measured M=1 region where every eligible shape reaches at least 1.10x median speedup in each of three independent count=7 campaigns; fallback cells must remain within 3%. Run full Metal and repository validation, retain negative variants as Spectackle evidence, and report generalizable findings to jxsl13/perfscan.

## T-01M0GX9710E94T4WT73GTTR3GZ Implement and gate aligned Q6_K ushort pairs
kind: task
state: active
created: 2026-08-21
parent: P-01M0GX86TYEEJSJ2SB873CJDDY
grilled: 2026-08-21 open=5
targets: msl:qmatmul_q6k_cooperative, objc:metal_bridge.mtl_qmatmul_q6k, go:metal.SetQ6KCooperative, go:metal.Recorder.QMatMulResident, backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/metal.go, backend/metal/q6k_bench_test.go

Implement a separately selectable qmatmul_q6k_cooperative_packed candidate. For each lane and Q6_K block, load q1, q2, and qh as six aligned ushort pairs instead of twelve indexed uchar values, then extract the identical bytes and preserve q6 reconstruction, scale indexing, and float accumulation order. Prove that the resident buffer base, 210-byte block stride, row stride, and all q offsets are two-byte aligned. Do not use wider pointer types.

Expose a toggle and shared route predicate used by direct, resident, and Recorder paths. Start default-off. Add scalar/control/candidate parity, planted non-finite classification, input immutability, support guard, M>1 fallback, and eventual threshold-boundary tests. Add an AB/BA resident benchmark covering KV, square, mid-up/down, gate/up, down, and vocabulary shapes with one transition dispatch excluded and 32 steady-state dispatches timed per arm.

Promote only if every eligible cell reaches at least 1.10x median speedup in each of three independent count=7 campaigns while fallback cells stay within 3%. Otherwise revert the experiment and preserve exact negative evidence. Run full repository, SIMD, and Metal validation, save reproducible evidence for any gain, and report generalizable findings to jxsl13/perfscan.
