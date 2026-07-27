---
schema: v1
prefix: MOE
---

## MOE-001
WHERE auxiliary-loss-free MoE routing is enabled, the router SHALL use the paper affinity exactly: s=sigmoid(routerLogit), TopK ranks s+b, and combine weights derive from s with b absent, renormalizing only after selection.

Rationale: The unoptioned Mixtral route stays softmax. The bias is control state outside Params and autograd and updates only through the explicit per-batch sign rule. Required gates include a decisive case where sigmoid(logit)+b and logit+b select different experts, plus a router finite-difference gradient check. Migrated from cavekit SPEC.md V32.

## LOSS-001
The EagleLoss feature regression SHALL compute mean SmoothL1 with beta=1 against the next feature, then add 0.1 times token cross-entropy, never substituting an MSE proxy.

Rationale: Required decisive gates: the abs(d)>1 linear branch, a full-head finite-difference gradient check, and lossless generation at an arbitrary head. Migrated from cavekit SPEC.md V33.

## OPT-001
WHERE Q-GaLore is selected, the optimizer SHALL apply all four deltas: block-wise INT8 moments, packed affine INT4 projection, adaptive SVD gap at 0.4/2/5, and INT8 weights re-quantized by unbiased stochastic rounding.

Rationale: QuantBits 0 or 64 disables all four and must stay bit-identical to GaLore across both projection sides and full rank. No end-to-end weight-memory claim is permitted while the public Tensor keeps a float execution mirror. Migrated from cavekit SPEC.md V34.

## OPT-002
WHERE APOLLO is selected, the optimizer SHALL persist only a two-word projection seed, low-rank moments and a limiter scalar, never retaining the O(r*min(m,n)) Gaussian projection matrix.

Rationale: P regenerates bit-identically from the seed on every use and changes only when the gap replaces that seed; reseeding preserves the moment buffers. Required gates cover both projection sides, seed regeneration equality, gap reseed difference, state shape, convergence and trajectory determinism. Migrated from cavekit SPEC.md V35.
