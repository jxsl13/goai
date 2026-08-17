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

## R-01M08XC0VNF9RVA1AG1HF28714 Reject M2 MPSGraph quant projection fusion despite broad-shape speedups
kind: research
state: draft
created: 2026-08-17
refs: P-01M08WKNGEF499G77J21EQA0BB, T-01M08WVXA2F6WTNZ1CRSH8Z4AG
targets: backend/metal/metal_bridge.m, backend/metal/metal_bridge.h, backend/metal/metal.go, backend/metal/dequant_gemm_bench_test.go

Base: merged main 85584691 on Apple M2 Pro. The disabled candidate cached MPSGraph objects by M,K,N, wrapped the existing recorder MTLCommandBuffer with MPSCommandBuffer, accepted f32 activations plus the persistent f16 Q4_K/Q6_K expansion, cast inside the graph, multiplied, and bound f32 output directly. The selector was live: candidate/control outputs differed on three shapes while both stayed finite. The declared numerical gate failed before end-to-end promotion. Initial max abs(diff)/max(1,abs(control)) was 2.747e-2 for Q4_K M64 K2048 N5632, 7.430e-1 for Q6_K M64 K5632 N2048, 9.882e-3 for Q4_K M24 K256 N513, and 0 for Q6_K M33 K512 N257, versus the 5e-4 limit. Explicitly forcing the graph result through f16 before widening left those errors unchanged. Expressing half-rounded operands through an f32 graph matmul and then storing f16 worsened the large Q6_K deviation to 1.024e0. The compiler lowering is therefore shape-dependent and not semantically equivalent to the current MPSMatrixMultiplication route. Ten alternating warmed leaf samples, three encoded operations per sample, showed a real but invalid performance ceiling: Q4_K K2048 N5632 477.5 versus 414.4 us, 1.152x; Q6_K K5632 N2048 599.5 versus 456.2 us, 1.314x; Q4_K K2048 N2560 251.8 versus 182.3 us, 1.381x; Q6_K K2048 N256 153.6 versus 153.7 us, 1.000x. No full pp64 or llama.cpp run was justified after parity failed. Every executable and test change was reverted. The generalized missing-parity guard is jxsl13/perfscan#761. Future graph experiments must prove reduction semantics at production K/N, not infer them from visible input/output dtypes or a small exact shape.
