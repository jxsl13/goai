# M2 Metal IQ1 evidence — 2026-08-22

Scope: Apple M2 Pro, resident single-token matrix-vector decode, exact GGUF IQ1_S
(wire type 19) and IQ1_M (wire type 29), F32 activations/accumulation, and the
Llama-family whole-token decoder. The base revision is
`f21a506578580a82eb0d40687d01bb04f9caa829`.

The candidate reconstructs the shared 2048x8 ternary codebook through
`gguf.Dequantize`, packs it to one uint16 per row, and retains one immutable
4 KiB Metal buffer. One 32-lane SIMD group owns one output row; every lane
decodes and dots one eight-value group. IQ1_S and IQ1_M retain independent wire
parsers and cooperative selectors.

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

- cooperative/scalar GPU floor: 4.332x
- cooperative/scalar recorder-wall floor: 3.085x
- resident Metal/fused ARM64 CPU floor: 1.967x
- unchanged scalar-control drift ceiling: 1.104x

The direct host-bound route is deliberately not promoted. Its measured ratios
are mixed and fall as low as 0.3324x CPU, despite isolated large-shape wins.
Generic `Backend.QMatMul` and `UploadQuant` therefore retain CPU fallback;
explicit resident recorder uploads use native Metal.

The captured whole-token run preserved identical greedy tokens and measured:

- IQ1_S: 55.51 to 233.62 tok/s = 4.209x
- IQ1_M: 66.38 to 278.04 tok/s = 4.189x

Correctness covered direct scalar/cooperative, resident recorder, M=1..4,
K=256..5632, Inf/NaN class, input immutability, invalid K, truncated weights,
and invalid dtype. The worst observed relative difference was 8.434e-6 against
the 1e-4 contract.

The pinned external perfscan v1.71.0 ratchet introduced zero findings against
the exact base: all 778 production and 1,811 test-inclusive findings were
suppressed by their respective baselines across `backend/metal`, `nlp`, and
`llamagpu`.

Raw campaign and environment files in this directory are the source of these
claims. `manifest.sha256` pins every evidence byte.
