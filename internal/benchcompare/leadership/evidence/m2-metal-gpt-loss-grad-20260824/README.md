# M2 Metal causal GPT objective evidence — 2026-08-24

## Claim cell

This evidence supports one narrow performance cell, not a universal framework
claim.

| Dimension | Value |
| --- | --- |
| Hardware | Apple M2 Pro, 32 GiB |
| Software | macOS 26.5.1 (25F80), Go 1.27.0 darwin/arm64 |
| Model | causal GPT, vocab/context/sequence 4096/256/256, width 512, 8 heads, FFN 2048, depth 6, batch 1 |
| Dtype/layout | contiguous offset-zero F32 |
| Objective | forward + mean hard-label cross-entropy + all 77 parameter gradients |
| Control | portable `Forward` + loss + private-tape `Backward` on Metal |
| Candidate | cached complete causal MPSGraph objective, one command-buffer submission |
| Boundary | warm execution; inputs and parameters already held by GoAI tensors |

The candidate is selected only for the exact supported model structure, dtype,
layout, objective, attention, and recorder state. Every other case retains the
portable implementation.

## Result

Three paired campaigns used the same test binary. Each campaign contains seven
aligned single-objective samples. Campaign two reverses which arm executes first.

| Campaign | Order | Median speedup | Minimum speedup |
| --- | --- | ---: | ---: |
| 1 | control first | 3.695x | 3.567x |
| 2 | candidate first | 3.620x | 3.447x |
| 3 | control first | 3.670x | 2.822x |
| All 21 samples | order-alternated | **3.689x** | **2.822x** |

The predeclared gate was aggregate median speedup of at least 1.25x and every
aligned sample at least 1.10x. All 21 samples pass. Across the accepted samples,
the candidate median was 20.748 ms/objective (12,338 tok/s) and the control
median was 76.238 ms/objective. A separate 5-objective allocation screen measured
245 allocations/objective and 12,444–12,648 tok/s for the candidate.

The pinned torch-mps comparison in `BENCHMARKS.md` is 12,904 tok/s for the same
geometry and objective boundary. This change therefore reduces the measured gap
from 3.95x to 1.046x; it does not claim a win over torch-mps.

## Correctness and reproducibility

The numerical tests compare the scalar loss and every parameter gradient,
exercise duplicate token indices and a sequence shorter than the context,
verify input and parameter immutability, exercise the portable F64 fallback,
and prove recorder isolation. The full Metal backend suite passes. See
`correctness.txt`, `samples.txt`, and `commands.txt`.

- Code commit: `fab7e6e5a129c2a12c268beb4aa5d59f2e7eef8d`
- Benchmark binary SHA-256:
  `3a2b9c1ad4c28ac5009a8eeaa9bd9e02d6200532042f7993688699852aa453f1`
- Spectackle research: `R-01M0S28QXJED6`
- Spectackle proposal: `P-01M0S2E3VVF8W`
- Spectackle decision: `ADR-01M0S2FBN8E06`
- Spectackle task: `T-01M0S2GVS8EB7`
- Generalized perfscan finding:
  <https://github.com/jxsl13/perfscan/issues/878>

## Leverage and rejected alternative

The useful accelerator boundary is the complete objective: embeddings, causal
transformer stack, loss, and reverse mode. It eliminates hundreds of synchronous
operation boundaries while keeping portable behavior at the public API.

An MPSGraph `scatterND` accumulation for repeated embedding indices was tested
and rejected. Short screens measured roughly 26.3–31.6 ms/objective versus
24.7–25.6 ms for dense one-hot transpose GEMM. The dense formulation remains in
the accepted graph; perfscan issue 878 records this shape-sensitive negative
result so future advice does not blindly prefer scatter.
