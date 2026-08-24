# M2 Metal ViT patch-sequence evidence (2026-08-24)

## Claim and scope

This evidence compares the composite control represented by merged `main` at
`9d0cf71c63783c05028a60f0c63d897163010f02` with implementation commit
`460b5bbc82890307b66ce8714f2e3d0a7397626a`. Both arms run from the same
binary on Go 1.27.0, macOS 26.5.1, and an Apple M2 Pro with 32 GiB unified
memory. The pinned F32 workload is ViT B=8, image 32x32, patch 4, sequence 65,
dimension 128, hidden dimension 512, depth 4, 4 heads, and 10 classes.

The control performs per-image Slice/Reshape/patch packing, concatenates the
patch rows, projects them once, then executes B Slice/Concat/Add sequence
assemblies and the composite VJP. The candidate directly packs the detached
image batch into its final patch matrix and executes patch projection, class
token broadcast, and position addition as one cached MPSGraph submission. Its
explicit backward graph produces gradients for patches, class token, position
embedding, weight, and bias in one submission. Reference F32/F64 kernels and
the original composite fallback preserve portability and unsupported layouts.

The Spectackle lifecycle is research `R-01M0RRGS0YEQ4`, proposal
`P-01M0RRS336FYB`, and task `T-01M0RRT1J6FSH`.

## Frozen gates

- Exact reference parity, Metal output/five-gradient parity, complete ViT
  logits and every parameter gradient, offset-layout fallback, and input
  immutability must pass.
- Three fresh-process, order-alternated count-seven campaigns per scope,
  `GOMAXPROCS=1`, one-second adaptive samples, and built-in warmups.
- Boundary median speedup at least 1.20x.
- Complete ViT training-step ratio-of-medians speedup at least 1.05x in every
  campaign.
- Every paired complete-step sample at least 1.03x.

## Result

Boundary campaign order is control/candidate, candidate/control, then
control/candidate. Complete-step samples interleave both arms inside each
calibrated iteration; the process-level leading arm follows the same order.

| Boundary campaign | Control median | Candidate median | Speedup |
|---|---:|---:|---:|
| 1 | 1.339105 ms | 0.673116 ms | 1.9894x |
| 2 | 1.322820 ms | 0.674492 ms | 1.9612x |
| 3 | 1.342286 ms | 0.677810 ms | 1.9803x |

| ViT campaign | Control median | Candidate median | Ratio of medians | Median paired speedup | Weakest pair |
|---|---:|---:|---:|---:|---:|
| 1 | 10.294687 ms | 9.653663 ms | 1.0664x | 1.067x | 1.058x |
| 2 | 10.232471 ms | 9.657506 ms | 1.0595x | 1.063x | 1.056x |
| 3 | 10.320916 ms | 9.797237 ms | 1.0535x | 1.066x | 1.053x |

All frozen gates pass. At the primitive training boundary, the candidate
reduces 6,687,336 to 437,240 B/op and 515 to 29 allocs/op: 6,250,096 bytes
and 486 allocations removed. At complete-step scope, it reduces 11,581,280 to
4,722,938 B/op and 1,119 to 329 allocs/op: 6,858,342 bytes and 790
allocations removed.

An earlier predecessor commit used grouped complete-step sub-benchmarks. Its
absolute latency drift eventually exceeded the roughly 6% end-to-end signal
and reversed one campaign, so that method was rejected rather than averaged
away. The final benchmark interleaves control and candidate within every
calibrated iteration while retaining grouped runs only for separate allocation
attribution.

## Generalized finding

Batch-indexed detached host preprocessing and a later device-side
Slice/Concat/Add loop can form one optimization boundary. Direct final-layout
packing plus a batch-aware forward/VJP op eliminates temporary host tensors,
backend dispatches, and native submissions together. The reusable detector
direction and these measurements are reported in
[perfscan issue 772](https://github.com/jxsl13/perfscan/issues/772#issuecomment-5390086434).

The exact benchmark binary is `/private/tmp/goai-patch-sequence-final.test`,
SHA-256 `cf74e04e097929bd17ff6d065064caa7f698bed1a20c600a03ad187ceb24ea41`.
Raw measurements, commands, correctness gates, and scanner results are stored
beside this file.
