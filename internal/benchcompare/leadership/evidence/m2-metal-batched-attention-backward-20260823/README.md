# M2 batch-axis attention backward evidence (2026-08-23)

## Claim and scope

This evidence compares merged `main` at `5d8afc8473c617fce501840a30bc9ac72eb0461c`
with the cached batch-axis Metal attention backward in this change. Both arms use
Go 1.27.0 on macOS 26.5.1 and an Apple M2 Pro. The pinned production workload is
ViT B=8, image 32x32, patch 4, sequence 65, dimension 128, depth 4, and 4 heads.

The control calls the synchronous MPSMatrix backward once per independent
sequence. The candidate feeds the packed B=8 tensors to one shape-keyed
MPSGraph, encodes once, waits once, and copies each packed gradient once.
Batch-one and sliding-window routes are unchanged.

## Frozen gates

- Three fresh-process, order-alternated count-seven M2 ViT B=8 campaigns.
- Training-step median speedup at least 1.10x in every campaign.
- Every aligned sample pair at least 1.05x.
- Causal and noncausal MHA/GQA/MQA gradient parity, including the production
  B=8/S=65/D=128/H=4 shape.

## Result

Each sample executes ten full forward, loss, and backward iterations. Campaign
order was control/candidate, candidate/control, then control/candidate.

| Campaign | Control median | Candidate median | Speedup | Worst aligned pair |
|---|---:|---:|---:|---:|
| 1 | 94.803 ms | 55.701 ms | 1.7020x | 1.5080x |
| 2 | 78.097 ms | 45.587 ms | 1.7131x | 1.6930x |
| 3 | 77.497 ms | 45.848 ms | 1.6903x | 1.6774x |

All frozen gates pass. The global worst aligned pair is 1.5080x. The first
campaign includes the expected fresh-process graph warm-up spread; keeping all
seven samples makes the reported result conservative.

## Correctness

- Metal batched backward matches the reference backend for causal/noncausal MHA,
  GQA, and MQA within the existing attention tolerances.
- The exact noncausal ViT B=8/S=65/D=128/H=4 shape passes forward and gradient
  cross-reference checks.
- Existing `Batch=0/1` code remains on the incumbent MPSMatrix path.
- A failed disposable autodiff formulation was removed after the installed
  MPSGraph runtime asserted while differentiating the broadcasted K path. The
  retained graph spells out the VJP and its GQA reductions explicitly.

Raw benchmark output and exact commands are stored beside this file.
