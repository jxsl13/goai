# M2 Metal Q1_0/MXFP4 evidence — 2026-08-22

Scope: Apple M2 Pro, resident single-token matrix-vector decode, exact GGUF
Q1_0 (wire type 41) and OCP MXFP4 (wire type 39), F32 activations and
accumulation, plus the Llama-family whole-token decoder. The exact base revision
is `5627d48d78f9c232987c96492a8cf2ebf73dbf60`.

The candidate compiles independent scalar and cooperative pipelines for the two
unrelated wire layouts. One 32-lane SIMD group owns one output row. Each Q1_0
lane reconstructs four binary values per 128-value block; each MXFP4 lane
reconstructs one split-half E2M1 nibble per 32-value block. Two SIMD groups share
one 64-thread threadgroup and produce two rows. The submission lifecycle is
shared, but neither decode hot loop contains a wire-type branch.

## Frozen representative cells

- KV: N=512, K=2048
- square: N=2048, K=2048
- gate/up: N=5632, K=2048
- down: N=2048, K=5632
- M=1 for every measured decode cell
- 16 distinct resident weights per cooperative sample
- 8 distinct host weights per direct-host sample

## Conservative promotion results

Floors and ceilings below are taken across the 12 campaign-by-cell medians from
three independent fresh-process count-seven campaigns.

| Format | GPU cooperative/scalar floor | recorder-wall floor | resident Metal/ARM64 CPU floor | scalar-control drift ceiling |
|---|---:|---:|---:|---:|
| Q1_0 | 2.643x | 2.017x | 1.574x | 1.034x |
| MXFP4 | 6.415x | 5.921x | 1.209x | 1.003x |

The equal-boundary direct host route is deliberately not promoted. Q1_0
campaign-by-cell medians ranged from 0.2641x to 1.037x CPU and MXFP4 ranged from
0.2618x to 0.8339x. Generic `Backend.QMatMul` and `UploadQuant` therefore
retain fused ARM64 CPU fallback; explicit Llama GPU uploads use resident Metal.

The final captured whole-token same-binary run preserved identical greedy
tokens:

- Q1_0: 101.27 to 223.34 tok/s = 2.205x
- MXFP4: 26.19 to 161.09 tok/s = 6.151x

Correctness covered independent GGUF cross-reference, forced scalar versus
cooperative execution, direct and recorder-resident paths, M=1..4,
K=32..5632, every MXFP4 nibble code including negative zero, Inf/NaN class,
input immutability, invalid K, truncated weights, invalid dtype, GGUF loader
admission, and exact Phi-3 row slicing. The worst observed relative difference
across the targeted suite was 1.345e-5 against the 1e-4 contract.

The pinned external perfscan v1.71.0 exact-base ratchet introduced zero findings:
778 production and 1,810 test-inclusive findings were suppressed. Module
resolution used `GOPROXY=direct`. The generalizable subgroup-per-output
evidence is recorded at jxsl13/perfscan#822, including the new evidence comment:
https://github.com/jxsl13/perfscan/issues/822#issuecomment-5381154017.

Raw campaign, correctness, whole-token, command, and environment files in this
directory are the source of these claims. `manifest.sha256` pins every evidence
byte.
