# M2 Metal IQ2_S evidence — 2026-08-22

Scope: Apple M2 Pro, resident single-token matrix-vector decode, exact GGUF
IQ2_S (wire type 22), F32 activations/accumulation, and the Llama-family
whole-token decoder. The base revision is
`e01131622d7a5bdf911cea86fd08c117695ef170`.

The candidate reconstructs the exact shared 1024x8 ternary codebook through
`gguf.Dequantize`, maps values 8, 25, and 43 to two-bit symbols, packs one
uint16 per row, and retains one immutable 2 KiB Metal buffer. One 32-lane SIMD
group owns one output row; every lane decodes and dots one eight-value group.
The IQ2_S wire parser and cooperative selector are independent from IQ2_XXS
and IQ2_XS.

## Frozen representative cells

- KV: N=512, K=2048
- square: N=2048, K=2048
- gate/up: N=5632, K=2048
- down: N=2048, K=5632
- M=1 for every measured decode cell
- 16 distinct resident weights per cooperative sample
- 8 distinct host weights per direct-host sample

## Conservative results

Across all four cells and all three independent fresh-process count-seven
cooperative campaigns:

- cooperative/scalar GPU floor: 4.278x
- cooperative/scalar recorder-wall floor: 3.204x
- resident Metal/fused ARM64 CPU floor: 2.302x
- unchanged scalar-control drift ceiling: 1.091x

The direct host-bound route is deliberately not promoted. Its measured ratios
are mixed and fall as low as 0.3492x CPU, while isolated large-shape wins do
not reproduce across fresh processes. Generic `Backend.QMatMul` and
`UploadQuant` therefore retain CPU fallback; explicit resident recorder
uploads use native Metal.

The captured whole-token same-binary run preserved identical greedy tokens and
measured 55.22 to 243.70 tok/s = 4.413x.

Correctness covered direct scalar/cooperative, resident recorder, M=1..4,
K=256..5632, Inf/NaN class, input immutability, invalid K, truncated weights,
and invalid dtype. The worst observed IQ2_S relative difference was 9.489e-6
against the 1e-4 contract.

The pinned external perfscan v1.71.0 ratchet introduced zero findings against
the exact base: all 778 production and 1,811 test-inclusive findings were
suppressed by their respective baselines across `backend/metal`, `nlp`, and
`llamagpu`. The generalizable resident-codebook optimization is tracked in
jxsl13/perfscan#821.

Raw campaign and environment files in this directory are the source of these
claims. `manifest.sha256` pins every evidence byte.
