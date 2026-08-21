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

## P-01M0GYKMR4E468TCHNX9G0930R Broadcast Q3_K scale headers once per SIMD group
kind: proposal
state: active
created: 2026-08-21
grilled: 2026-08-21 open=1
targets: msl:qmatmul_q3k_cooperative, objc:metal_bridge.ensure_qmatmul_q3k, objc:metal_bridge.mtl_qmatmul_resident, objc:metal_bridge.mtl_recorder_qmatmul, objc:metal_bridge.mtl_qmatmul_q3k, go:metal.SetQ3KCooperative, backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/metal.go, backend/metal/q3k_cooperative_test.go

The cooperative Q3_K decode kernel currently rebuilds the same 12-byte packed scale header independently in all 32 SIMD lanes for every 256-value superblock, even though each lane consumes only one of the 16 unpacked scales. Test a separately selectable qmatmul_q3k_cooperative_broadcast candidate on Apple M2 Pro: lane zero loads the header as six alignment-safe ushort values, combines them into aux0/aux1/aux2, broadcasts those three words with simd_broadcast_first, and preserves the existing per-lane scale splice, q3 reconstruction, inverted hmask semantics, and accumulation order. Q3_K blocks and rows are 110-byte aligned only to two bytes, so no uint or wider device-pointer loads are allowed. Wire one default-off predicate through direct, resident, and Recorder selectors. Validate scalar/control/candidate parity, finite and nonfinite classes, immutability, odd tails, support guards, and M>1 fallback. Benchmark same-binary AB/BA resident decode across KV, square, mid-up/down, gate/up, down, and vocabulary shapes with transition dispatches excluded. Promote only if every eligible production shape reaches at least 1.10x control in each of three independent count-seven campaigns; otherwise revert and preserve negative evidence.

## T-01M0GYMW10ETFB63R2WSAJB16E Implement and gate Q3_K SIMD scale broadcast
kind: task
state: active
created: 2026-08-21
parent: P-01M0GYKMR4E468TCHNX9G0930R
grilled: 2026-08-21 open=5
targets: msl:qmatmul_q3k_cooperative, objc:metal_bridge.ensure_qmatmul_q3k, objc:metal_bridge.mtl_qmatmul_resident, objc:metal_bridge.mtl_recorder_qmatmul, objc:metal_bridge.mtl_qmatmul_q3k, go:metal.SetQ3KCooperative, backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/metal.go, backend/metal/q3k_cooperative_test.go, backend/metal/kquant_gap_bench_test.go

Implement a separately selectable qmatmul_q3k_cooperative_scale_broadcast candidate. In every superblock, lane zero shall read the 12-byte Q3_K scale header through six ushort loads, combine them into aux0/aux1/aux2, and broadcast those three values with simd_broadcast_first. All lanes shall retain the historical a0/a1/a2/a3 splice, scv selection, q3 and inverted hmask reconstruction, and float accumulation order. Prove two-byte alignment from the resident base, 110-byte block and row strides, 96-byte scale offset, and half-d offset; never cast device memory to uint or wider. Add a default-off toggle and one shared direct/resident/Recorder route predicate. Add scalar/control/candidate parity across K=256 through K=4096, odd N, planted NaN and infinity classification, input immutability, support gating, M greater than one fallback, and candidate-disabled checks. Add a same-binary AB/BA resident benchmark for KV, square, mid-up/down, gate/up, down, and vocabulary shapes, excluding one transition dispatch and timing 32 steady dispatches per arm. Run a three-sample pilot. Promote only if every eligible shape reaches at least 1.10x in each of three independent count-seven campaigns; otherwise revert and preserve exact negative evidence. Run full repository, SIMD, and Metal validation, save reproducible evidence, and report generalizable findings to jxsl13/perfscan.
