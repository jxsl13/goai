# M2 Metal ViT objective evidence — 2026-08-24

## Claim cell

This evidence supports one deliberately narrow leadership cell; it is not a
claim that GoAI is universally faster than another framework.

| Dimension | Value |
| --- | --- |
| Hardware | Apple M2 Pro, 32 GiB |
| Software | macOS 26.5.1 (25F80), Go 1.27.0 darwin/arm64 |
| Model | ViT, batch 8, sequence 65, width 128, 4 heads, FFN 512, depth 4, 10 classes |
| Dtype/layout | contiguous offset-zero F32 |
| Objective | forward + mean basic cross-entropy + all 56 parameter gradients |
| Control | portable `Forward` + loss + private-tape `Backward` |
| Candidate | cached complete MPSGraph objective, one command-buffer submission |
| Boundary | warm execution; inputs and parameters already resident in GoAI tensors |

The candidate preserves the portable implementation as its fallback. Metal is
selected only for the supported geometry, dtype, layout, objective, attention,
and recorder state.

## Result

Three paired campaigns used the exact same test binary. Each campaign contains
seven samples and each sample runs 20 objectives per arm. Campaign two reverses
the arm order.

| Campaign | Order | Median speedup | Minimum speedup |
| --- | --- | ---: | ---: |
| 1 | control first | 1.454x | 1.439x |
| 2 | candidate first | 1.450x | 1.436x |
| 3 | control first | 1.499x | 1.321x |
| All 21 samples | alternating | 1.456x | 1.321x |

The predeclared gate was a median of at least 1.20x in every campaign and at
least 1.10x in every sample. All three campaigns pass.

An earlier short screen was invalidated before publication because unrelated
multi-core perfscan validation was active and produced a scheduler outlier. No
sample from that screen is included here. The raw accepted samples are in
`samples.txt`.

## Correctness and reproducibility

The numerical tests compare the scalar loss and every parameter gradient,
verify input and parameter immutability, exercise the portable F64 fallback,
and prove recorder isolation. The final Metal preflight passes. See
`correctness.txt` and `commands.txt`.

- Code commit: `2f95fa87db1e5e9af037751dc2031488c08dc5b1`
- Benchmark binary SHA-256:
  `4fa0f337813ef4b5eedfb0e4e3001eb7ccdcb5cd0014d21ab822bfcf28e22855`
- Spectackle research: `R-01M0RX17A7E98`
- Spectackle proposal: `P-01M0RX1WNKETX`
- Spectackle task: `T-01M0RX4KE4E11`
- Generalized perfscan finding:
  <https://github.com/jxsl13/perfscan/issues/876>

## Leverage

The generalizable improvement is to recognize a model objective as the useful
accelerator boundary. Exporting an activation after a fused forward and then
re-entering the backend for loss and reverse mode can repeat graph work and
synchronization. A narrow optional whole-objective capability removes that tax
without weakening the portable API or widening the primitive backend contract.

