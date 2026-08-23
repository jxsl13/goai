# M2 batch-axis attention evidence (2026-08-23)

## Claim and scope

This evidence compares `origin/main` at `3eac08330e890d87ba1fe4c8baf2adceb191b98d`
with batch-aware `OpMHA` on an Apple M2 Pro. Both arms use Go 1.27.0 on macOS
26.5.1. The pinned workload is ViT B=8, image 32x32, patch 4, sequence 65,
dimension 128, depth 4, and 4 heads.

The control packs projection GEMMs but executes 3B slices, B attention graphs,
and one concat per attention core. The candidate represents B independent
sequences as a graph axis and executes one attention graph. Batch-one behavior is
unchanged; reference, CPU, CUDA, and Vulkan retain correct portable semantics.

## Frozen gates

- Leaf B=8/S=65/D=128/H=4: median GPU and wall speedup >=1.50x; every pair
  >=1.25x; existing Metal numerical tolerance.
- End-to-end M2 ViT B=8: forward median >=1.15x, train-step median >=1.10x,
  and every aligned pair >=1.05x.
- Batch-one compatibility, batch isolation, inference parity, and gradient parity.

## Leaf result

A disposable same-binary native probe compared eight warmed incumbent MPSGraph
calls with one warmed batch-axis graph over seven alternating pairs, twelve
repetitions per arm.

- wall median: 3.4453x; minimum: 3.1841x
- reported command GPU median: 49.4942x; minimum: 44.8235x
- max absolute and relative output error: 0

The GPU timestamp magnitude indicates that MPSGraph command timestamps omit
meaningful scheduling or transfer work, so wall time is the primary leaf proof.
The disposable probe was removed after the production path and tests existed.

## End-to-end result

Each campaign contains seven benchmark samples with ten complete iterations per
sample. Campaign order was control/candidate, candidate/control, then
control/candidate.

| Campaign | Forward control | Forward candidate | Throughput gain | Train control | Train candidate | Throughput gain |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 21.39 ms | 14.30 ms | +49.55% | 85.73 ms | 74.86 ms | +14.55% |
| 2 | 21.20 ms | 14.00 ms | +51.43% | 86.33 ms | 74.59 ms | +15.79% |
| 3 | 23.22 ms | 14.45 ms | +60.75% | 88.52 ms | 73.76 ms | +20.05% |

Worst aligned pair across the campaigns was 1.4292x forward and 1.1105x train.
All frozen gates pass. A three-sample CPU smoke comparison also improved median
forward from 35.35 ms to 20.48 ms and train-step from 114.88 ms to 83.00 ms; it
is supporting evidence, not a confidence-qualified claim.

The `BENCHMARKS.md` table uses its documented `GOEXPERIMENT=simd` mode. A separate
count-seven candidate run measured median 1,371/472 img/s for CPU forward/train
and 602/104 img/s for Metal forward/train; see `candidate-simd.txt`.

## Correctness and tooling

- Reference F64 batched forward/backward is bit-identical to independent calls.
- CPU F32/F64 forward/backward matches reference within the existing ULP budget.
- Metal batched causal/noncausal forward and backward match reference within the
  existing attention tolerances; legacy batch-one Metal cases remain green.
- ViT batched logits retain exact F64 parity with per-image execution.
- External `github.com/jxsl13/perfscan/perfscan@latest` ran with `GOPROXY=direct`;
  the production diff has zero new findings.
- The generalizable dispatch pattern is recorded in
  `jxsl13/perfscan#772` (comment `issuecomment-5387979330`).

Raw benchmark output and exact commands are stored beside this file.
