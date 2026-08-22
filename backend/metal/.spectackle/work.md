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

## P-01M0M9B6FRFCZA18408PMM2WGH Add native M2 Metal Q4_1 quantized matmul
kind: proposal
state: active
created: 2026-08-22
grilled: 2026-08-22 open=0
targets: go:metal.QMatMulQ4_0, backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/qmatmul_test.go

Complete the bottom-up Q4_1 path by adding exact GGUF type-3 dispatch to the Metal backend. Reuse the proven Q4_0 scalar and SIMD-group shapes but decode each 20-byte block as f16 d, f16 m, and 16 split-half nibbles with value d*q+m. Cover per-call, resident, and recorder dispatch; preserve explicit unsupported boundaries elsewhere. Validate against gguf.QMatMul, forced scalar/cooperative parity, invalid-input rejection, and M2 warm/cold benchmarks. Keep the change only if it beats CPU fallback end to end in a declared leadership cell without weakening semantics.

## T-01M0M9BPF2FW68JAGG2XZ8F917 Implement and benchmark native Metal Q4_1
kind: task
state: draft
created: 2026-08-22
parent: P-01M0M9B6FRFCZA18408PMM2WGH
grilled: 2026-08-22 open=1
targets: go:metal.QMatMulQ4_0, backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/qmatmul_test.go

Add GGUF type-3 constants and dispatch, synchronous and resident/recorder execution, an exact 20-byte affine Q4_1 Metal decoder, SIMD-group cooperative M2 path, correctness and scope tests, and same-binary/per-process benchmark evidence. Validate full repository, external perfscan, and retain only measured leverage.
