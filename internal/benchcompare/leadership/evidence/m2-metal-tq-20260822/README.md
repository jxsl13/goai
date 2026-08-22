# M2 Metal TQ1_0/TQ2_0 evidence — 2026-08-22

Scope: Apple M2 Pro, resident single-token matrix-vector decode, exact GGUF
TQ1_0 (wire type 34) and TQ2_0 (wire type 35), F32 activations/accumulation,
and the Llama-family whole-token decoder. The base revision is
`04838955aa12ce709c46f5a3cea22446534b7cb4`.

The candidate adds independent scalar controls and SIMD-group cooperative
kernels for both formats. One 32-lane SIMD group owns one output row and each
lane reconstructs exactly eight values per 256-value block. TQ1_0 balances
base-243 and tail encodings through logical positions `lane + 32*j`; TQ2_0
maps each lane directly onto its bit-plane codes. Two SIMD groups share one
64-thread threadgroup and produce two output rows. No qtype branch appears in
either format's hot loop.

## Frozen representative cells

- KV: N=512, K=2048
- square: N=2048, K=2048
- gate/up: N=5632, K=2048
- down: N=2048, K=5632
- M=1 for every measured decode cell
- 16 distinct resident weights per cooperative sample
- 8 distinct host weights per direct-host sample

## Conservative results

Across both formats, all four cells, and all three independent fresh-process
count-seven cooperative campaigns:

- cooperative/scalar GPU floor: 11.99x
- cooperative/scalar recorder-wall floor: 7.771x
- resident Metal/fused ARM64 CPU floor: 2.056x
- unchanged scalar-control drift ceiling: 1.005x

The direct host-bound route is deliberately not promoted. Its equal-boundary
ratios are mixed and fall as low as 0.3155x CPU. Generic
`Backend.QMatMul` and `UploadQuant` therefore retain CPU fallback; explicit
resident recorder uploads use native Metal.

The captured whole-token same-binary run preserved identical greedy tokens:

- TQ1_0: 9.03 to 273.53 tok/s = 30.288x
- TQ2_0: 33.71 to 379.62 tok/s = 11.261x

Correctness covered direct scalar/cooperative, resident recorder, M=1..4,
K=256..5632, Inf/NaN class, TQ2 raw code 3 as +2, input immutability, invalid
K, truncated weights, invalid dtype, GGUF loader admission, and exact Phi-3
row slicing. The worst observed relative difference was 2.354e-6 against the
1e-4 contract.

The pinned external perfscan v1.71.0 ratchet introduced zero findings against
the exact base: 778 production and 1,810 inherited test-inclusive findings
were suppressed in the candidate run. The exact-base test baseline contained
1,811 findings; one inherited test-only allocation was explicitly documented
and suppressed by the candidate. The generalizable subgroup-per-output
optimization is tracked in jxsl13/perfscan#822.

Raw campaign and environment files in this directory are the source of these
claims. `manifest.sha256` pins every evidence byte.
