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

## P-01M0QRX6E0FC1S4XNAWGQK8900 Autotune M2 Q4_K output rows per SIMD group
kind: proposal
state: active
created: 2026-08-23
refs: R-01M0QRNR7FF1TT618GVH65BMHS
grilled: 2026-08-23 open=0
targets: msl:qmatmul_q4k_cooperative, objc:metal_bridge.mtl_recorder_qmatmul, objc:metal_bridge.mtl_qmatmul_resident

The production stage profile attributes 4.856 ms across 110 Q4_K events, and the retained cooperative kernel uses two output rows per SIMD group. Pinned llama.cpp 4a08fa29705b8177e332b134306566c2c4b95902 retains the same N_R0_Q4_K=2 default, while its issue 19303 reports output-row sensitivity on M5. Prior GoAI M2 work swept Q6_K rows, Q4_K threadgroup widths, and alternative Q4_K kernels, but not the retained float-activation Q4_K rows-per-SIMD parameter. Add diagnostic-only 4-row and 8-row pipelines without altering the exact 2-row control, select them through a same-binary test toggle, and preserve identical per-row arithmetic. Freeze gates: exact candidate/control bits and input immutability across odd and production shapes; control pipeline within 2 percent of the unmodified baseline; each retained candidate at least 1.05x on K2048/N2048, K2048/N5632, and K5632/N2048 in every one of three alternating count-seven campaigns; production TinyLlama decode at least 1.02x with identical token/logit digests and no stage regression outside Q4_K. Reject and remove all executable changes if no variant clears every gate.

## T-01M0QRY7EBEA99HWV2A0NKWTY1 Implement and gate Q4_K rows-per-SIMD variants
kind: task
state: active
created: 2026-08-23
parent: P-01M0QRX6E0FC1S4XNAWGQK8900
targets: msl:qmatmul_q4k_cooperative, objc:metal_bridge.mtl_recorder_qmatmul, objc:metal_bridge.mtl_qmatmul_resident

Add diagnostic-only 4-row and 8-row Q4_K cooperative pipelines while preserving the production 2-row source and selector as the control. Expose a test-only same-binary selector, derive recorder and resident dispatch geometry from the selected rows, assert toggle arms differ, and add exact odd-tail plus production-shape parity/input-immutability coverage. Run three order-alternated count-seven campaigns at K2048/N2048, K2048/N5632, and K5632/N2048. Promote only a variant clearing 1.05x in every leaf cell and 1.02x in every exact TinyLlama decode campaign with identical token/logit digests; otherwise revert all executable code and reject with measurements.
