# M2 pre-norm FFN graph-fusion evidence (2026-08-23)

## Claim and scope

This evidence compares merged `main` at
`428b7b4df0029f56a0eb29d6db83e88a123b5098` with a fused pre-norm exact-GELU
FFN training boundary. Both arms use Go 1.27.0 on macOS 26.5.1 and an Apple M2
Pro. The pinned production workload is ViT B=8, image 32x32, patch 4, sequence
65, dimension 128, hidden dimension 512, depth 4, and 4 heads.

The control executes LayerNorm, MatMul, AddBias, GELU, MatMul, AddBias, and the
residual Add independently. The candidate uses one bounded shape-keyed
MPSGraph submission for forward and one explicit-VJP submission for backward.
Epsilon remains a runtime scalar input, and unsupported backends, dtypes, or
layouts retain the seven-operation composite.

## Frozen gates

- Output, all seven input gradients, and input immutability must match control.
- Three fresh-process, order-alternated count-seven campaigns per benchmark.
- Production boundary median speedup at least 1.20x.
- ViT B=8 training median speedup at least 1.10x in every campaign.
- Every aligned ViT sample pair at least 1.05x.

## Result

Each boundary sample executes twenty complete forward/backward boundaries. Each
ViT sample executes ten full forward, loss, and backward steps. Campaign order
is control/candidate, candidate/control, then control/candidate.

| Boundary campaign | Control median | Candidate median | Speedup | Worst aligned pair |
|---|---:|---:|---:|---:|
| 1 | 6.106 ms | 1.859 ms | 3.2850x | 2.8664x |
| 2 | 5.516 ms | 1.903 ms | 2.8993x | 2.1078x |
| 3 | 6.177 ms | 2.144 ms | 2.8817x | 1.8671x |

| ViT campaign | Control median | Candidate median | Speedup | Worst aligned pair |
|---|---:|---:|---:|---:|
| 1 | 75.968 ms | 56.510 ms | 1.3443x | 1.1939x |
| 2 | 82.679 ms | 60.482 ms | 1.3670x | 1.1553x |
| 3 | 140.013 ms | 103.051 ms | 1.3587x | 1.2363x |

All frozen gates pass. Absolute latency shifts under sustained thermal load in
campaign 3, while the speedup remains 1.3587x; the gain therefore survives the
clock window rather than depending on it.

## Correctness and static analysis

- The F64 reference operation matches the seven-operation reference composite.
- The target Metal boundary matches output, dX, dGamma, dBeta, dW1, dB1, dW2,
  and dB2 within the established F32 tolerance and mutates zero inputs.
- The complete candidate ViT matches unfused Metal logits and every parameter
  gradient.
- `go test -short ./...`, `go vet ./...`, and Spectackle task-drift checks pass.
- Upstream perfscan v1.81.1 from GitHub reports identical full-tree counts:
  1,959 production and 6,079 including tests, with zero changed-file findings.
- The generalizable detector opportunity is tracked in
  [perfscan issue 870](https://github.com/jxsl13/perfscan/issues/870).

Raw measurements, exact commands, and validation output are stored beside this
file.
