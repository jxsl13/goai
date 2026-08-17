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

## T-01M08TYB09FY9BCTHEVDRGQPE9 Implement and gate the M2 Q4_K 64x32 legacy simdgroup tile
kind: task
state: active
created: 2026-08-17
parent: P-01M08TT3TTE7ZBJCE64TC94G5C
grilled: 2026-08-17 open=0
targets: backend/metal/metal_bridge.m, backend/metal/q4k_mm_parity_test.go, backend/metal/q4k_mm_crossover_test.go

Replace the disabled qmatmul_q4k_mm experiment with a faithful GoAI-oriented adaptation of the pinned llama.cpp b10450 legacy Q4_K matrix-matrix geometry: output tile M=32 rows by N=64 channels, K=32 iteration, four simdgroups, 4096-byte half Q4_K weight staging, 2048-byte half f32-input staging, eight float accumulators per simdgroup, direct full-tile f32 stores, and guarded partial-tile writeback. Preserve Q4_K block bytes and dequantization semantics, row-major X[M,K], W[N,K], O[M,N], the public test toggle, and all production selectors for the first pass.

Update the existing matrix-unit parity test to exercise the new N=64 boundary and partial M/N tiles, reject non-finite results before aggregate error, compare against cached-f16 production semantics on a trained-like target shape, and prove the selector is live by mutation or profile label. Replace the old expected-rejection crossover guard only if the complete geometry invalidates it.

Run ten alternating warm samples against the actual cached-f16 MPS path at M64 K2048 N5632 plus M32/48/64/96/128/256. Require >=1.10x at the target and no routed shape below 0.98x before changing production selection. Only after leaf promotion, run ten alternating fresh-decoder TinyLlama Q4_K_M pairs with pp64 >=1.03x, pp128/pp512/tg64 >=0.98x, finite logits, identical greedy argmax, and unchanged steady-state allocations. Fully revert executable and test changes on any failed gate, archive exact evidence, and report generalizable findings to jxsl13/perfscan.
