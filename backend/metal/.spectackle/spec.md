---
schema: v1
---

## intent
- R-01M08T14CKES181P11RG303W1H M2 Q4_K half-MMA improves the old prototype but is 0.4018x of cached-f16 MPS: Consumed by rejection of T-01M08SPZPNEG6 and P-01M08SN9PFECD. The half-input float-accumulate Q4_K simdgroup candidate preserved finite trained-like parity (max relative error 2.947e-4) but measured 960.125 us versus 385.750 us for the production cached-f16 MPS path at M64 K2048 N5632, only 0.4018x baseline throughput and far below the 1.10x promotion gate. All executable and test changes were ful [body truncated at tombstone retention cap]
- R-01M08VQWMFEB9S9JCJ8NVSTYC2 Reject the M2 legacy Q4_K 64x32 tile against the production cache: Closed no-action after the leaf gate failed. The complete pinned llama.cpp legacy tile was finite but missed the declared cached-f16 numerical gate and reached only 0.741x/0.746x production throughput at M64 in two fresh processes. The old uncached-f32 comparator would have reported a false win. The executable experiment was fully reverted; the generalized second-reuse-axis detector is tracked in [body truncated at tombstone retention cap]
- R-01M08XC0VNF9RVA1AG1HF28714 Reject M2 MPSGraph quant projection fusion despite broad-shape speedups: Closed no-action and consumed by the rejection of T-01M08WVXA2F6W and P-01M08WKNGEF49. MPSGraph delivered 1.152x-1.381x on broad M64 projection leaves but changed reduction results far beyond the 5e-4 gate; two explicit rounding/accumulation repairs failed. All executable changes were reverted. The generalized parity-before-timing guard is jxsl13/perfscan#761.
- T-01M0FNQVFTFQ0ABX96T9QM18R4 Gate deterministic host Metal embedding backward: Implemented deterministic host-resident F32 scatter with reference-order rounding; preserved the old Metal atomic route as a mutation-proven benchmark control. Three independent seven-sample M2 campaigns cleared all five frozen shapes, worst 3.931x and best 30.762x. Cross-reference, preflight, full Metal lane, focused perfscan, and all 15 PR #1105 CI checks passed. Evidence: internal/benchcompare/ [body truncated at tombstone retention cap]
- T-01M0FVGM88EWMRQCHFN4B748AV Gate Metal bias-gradient routing against the optimized CPU kernel: Add a mutation-proven direct-Metal benchmark control and production selector over the frozen F32 shape matrix. Run three independent count-7 campaigns, reject unstable timing before routing, pin both selector arms, strengthen contiguous and noncontiguous reference parity, run an end-to-end GPT training-step no-regression gate, and retain only a measured winner zone. Record reproducible evidence an [body truncated at tombstone retention cap]

## METAL-RESIDENT-TOPK-001
WHEN TopKN is called with valid n and k on a live f32 DeviceBuffer, the Metal resident selection boundary SHALL return k distinct first-n index/value pairs matching the host top k, ordered by descending value then ascending index.

Rationale: Exact deterministic selection preserves the existing sampler draw.

## METAL-RESIDENT-TOPK-TRANSFER-001
WHEN TopKN returns k candidates from n resident logits, the Metal sampling boundary SHALL copy and allocate O(k) result data in Go without materializing the n logits.

Rationale: The optimization exists to eliminate the measured full-vocabulary host boundary.

## HOST-RESIDENT-EMBED-BACKWARD-001
WHEN receives a valid host-resident F32 table, index vector, and upstream gradient, the Metal OpEmbedBackward SHALL execute exactly 0 Metal command submissions and return the deterministic reference-order scatter-add gradient.

## MEASURED-METAL-BIAS-GRAD-ROUTE-001 {applies: go:metal.addBiasBackwardF32}
WHEN a synchronous host-resident F32 bias gradient is requested, the Metal add-bias backward SHALL route through CPU only where 3 count-7 campaigns each prove at least 1.10x median speedup, and preserve direct Metal elsewhere.

Rationale: A later exact CPU reduction can invalidate an older synchronous GPU route, but the winner zone must remain measurement-bounded.

## MEASURED-METAL-BIAS-ROUTE-001
WHEN a synchronous host-resident F32 bias add is requested, the Metal host wrapper SHALL route through CPU only where 3 count-7 campaigns each prove at least 1.10x median speedup and preserve direct Metal elsewhere.

Rationale: A later optimized CPU broadcast kernel can invalidate an older synchronous GPU route, but the winner zone must remain measurement-bounded.

## METAL-Q5K-WIDE-LOAD-PERF-001
WHEN a resident M=1 Q5_K vector-load kernel is considered for production on Apple M2, the Metal Q5_K selector SHALL retain it only when every representative shape in three independent count-7 same-binary campaigns reaches at least 1.10x the current cooperative kernel.

## METAL-Q5K-WIDE-LOAD-NUMERIC-001
WHEN it executes a valid resident M=1 projection, the Metal Q5_K vector-load kernel SHALL match the current cooperative output within 2e-5 relative error across K=256 through K=4096, preserve NaN class, and leave activation and weight inputs unchanged.

## METAL-Q5K-WIDE-LOAD-SCOPE-001
WHEN M is greater than 1 or the device lacks a 32-lane SIMD group with 64-thread threadgroups, the Metal Q5_K selector SHALL execute the existing Q5_K path with zero vector-load-candidate dispatches.

## METAL-Q5K-WIDE-LOAD-THRESHOLD-001
WHEN a Q5_K cooperative pipeline is selected, the Metal Q5_K selector SHALL dispatch the aligned per-lane uint2 vector-load candidate only when M equals 1 and K times N is at least 6291456; otherwise dispatch the historical cooperative pipeline.

Rationale: The steady-state M2 boundary sweep measured 1.117x at 2048x3072 but only 1.055x at 2048x2048; per-lane coalesced uint2 loads also beat the simd_shuffle sharing variant on eligible FFN cells.

## METAL-Q4K-WIDE-LOAD-PERF-001
WHEN a resident M=1 Q4_K vector-load candidate is considered for production on Apple M2, the Metal Q4_K selector SHALL retain it only when every eligible shape in three independent count-7 same-binary campaigns reaches at least 1.10x the cooperative kernel.

Rationale: A packed-load microbenchmark is promotable only as a repeatable shape-bounded production win.

## METAL-Q4K-WIDE-LOAD-NUMERIC-001
WHEN it executes a valid resident M=1 projection, the Metal Q4_K vector-load kernel SHALL match the current cooperative output within 2e-5 relative error, preserve finite Inf and NaN classification, and leave activation and weight inputs unchanged.

Rationale: Changing load width must not change Q4_K bytes, arithmetic semantics, exceptional-value behavior, or ownership.

## METAL-Q4K-WIDE-LOAD-SCOPE-001
WHEN M is greater than 1 or the device lacks a 32-lane SIMD group with 64-thread threadgroups, the Metal Q4_K selector SHALL execute the existing Q4_K path with zero vector-load-candidate dispatches.

Rationale: The experiment is an M2 M=1 specialization and must retain the portable historical fallback.
