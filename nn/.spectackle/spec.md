---
schema: v1
prefix: MOE
---

## MOE-002
WHERE auxiliary-loss-free MoE routing is enabled, the router SHALL use the paper affinity exactly: s=sigmoid(routerLogit), TopK ranks s+b, and combine weights derive from s with b absent, renormalizing only after selection.

Rationale: The unoptioned Mixtral route stays softmax. The bias is control state outside Params and autograd and updates only through the explicit per-batch sign rule. Required gates include a decisive case where sigmoid(logit)+b and logit+b select different experts, plus a router finite-difference gradient check. Migrated from cavekit SPEC.md V32.

## LOSS-002
The EagleLoss feature regression SHALL compute mean SmoothL1 with beta=1 against the next feature, then add 0.1 times token cross-entropy, never substituting an MSE proxy.

Rationale: Required decisive gates: the abs(d)>1 linear branch, a full-head finite-difference gradient check, and lossless generation at an arbitrary head. Migrated from cavekit SPEC.md V33.

## OPT-003
WHERE Q-GaLore is selected, the optimizer SHALL apply all four deltas: block-wise INT8 moments, packed affine INT4 projection, adaptive SVD gap at 0.4/2/5, and INT8 weights re-quantized by unbiased stochastic rounding.

Rationale: QuantBits 0 or 64 disables all four and must stay bit-identical to GaLore across both projection sides and full rank. No end-to-end weight-memory claim is permitted while the public Tensor keeps a float execution mirror. Migrated from cavekit SPEC.md V34.

## OPT-004
WHERE APOLLO is selected, the optimizer SHALL persist only a two-word projection seed, low-rank moments and a limiter scalar, never retaining the O(r*min(m,n)) Gaussian projection matrix.

Rationale: P regenerates bit-identically from the seed on every use and changes only when the gap replaces that seed; reseeding preserves the moment buffers. Required gates cover both projection sides, seed regeneration equality, gap reseed difference, state shape, convergence and trajectory determinism. Migrated from cavekit SPEC.md V35.

## intent
- T-01KYJR5WRXF5CSDZC316FB10T5 Replace Muon's private naive GEMM — it runs at 3.3 GFLOP/s beside the repo's own 61 GFLOP/s kernel: LANDED steps 2+3 of the four-part fix, benchmark-validated on M2 Pro darwin/arm64 go1.26.5: matmulABt rewritten from dot-per-output to ikj/axpy with a one-shot k-dim transpose, plus the symmetry halving available because newtonSchulz5 calls it as matmulABt(X,X,...). BenchmarkMuonStepOnly 418.3 -> 200.0 ms median (2.09x), 3 reps of 30x each, interleaved with BenchmarkAdamStepOnly as an unaffected control. This matches the brief predicted ~200 ms for steps 1-3 exactly. Bytes/op 26.7 -> 28.3 MB (+6%): the transpose scratch is hoisted to once per newtonSchulz5 run rather than per iteration, which recovered most of the 34.6 MB the naive placement cost.

BIT-IDENTITY PROVEN, NOT ASSUMED, and the proof needed strengthening: the first cross-reference gate caught a reversed accumulation order but NOT the insertion of matmulFlat zero-skip, because random fixtures contain no exact zero. That skip is not order-preserving (it drops 0*(+-Inf) NaNs), so an explicit zero/Inf fixture was added and the rewrite deliberately omits the skip. Both mutations now turn the gate red.

GENERALIZED as perfscan PS4008 serial-dot-matmul with 5 fixtures and a mutation-probed detector: 26 candidates tree-wide, and it independently rediscovered all 8 sites of the hand-found SOAP/Shampoo basis-rotation task. Silent on the ikj/axpy form itself, on same-base reductions with no indexed store, and on compound inner bodies.

NOT DONE, deliberately, both split out as follow-ups: step 1 (hoist bm/A/A2/bx onto the Muon struct, which is what would take bytes/op BELOW baseline rather than 6% above) and step 4 (route the three products through ops.MatMul to reach the parallel gemmF64Band, the predicted further ~5x). Step 4 remains unvalidated here because it changes the kernel rather than the loop order, so its bit-identity must be measured against the tolerance-0 tests rather than argued.

Pre-existing unrelated red in the package: TestEMAUpdateBitIdenticalToSlowPath fails identically before and after.
