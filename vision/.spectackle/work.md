---
schema: v1
---

## T-01KYJPMNDEE14TWT5FEKCQJ401 Batch the Swin permutation gathers and memoize their index tensors
kind: task
state: draft
created: 2026-07-27

LAND THIS FIRST. It is the only fully bit-identical item in the Swin set and it de-risks the fused-attention task, which edits the same function.

SITE: vision/swin.go:382-395 and :413-427 in (*SwinBlock).forwardBatched; helpers swinGather (:819), swinIdxTensor (:825), swinShiftIdx (:871), swinPartitionIdx (:842), swinInverse (:860), swinBuildMasks (:907).

DEFECT, two compounding parts.
(a) The window permutation is image-independent but applied per image. shiftIdx and partIdx are computed once at :378-379, then the loop at :382 issues one OpSlice plus one or two swinGather calls per image. Because the batched packing is b*N + i, a single index array of length batch*N (entry b*N + perm[i]) collapses B*(1 slice + 2 gathers) into ONE OpEmbed over the whole packed tensor. Same on the inverse side at :413-427. At batch 8 that is roughly 48 dispatches per block reduced to 4, so about 192 to 16 across the four blocks of the pinned config.
(b) The index tensors are rebuilt inside the loop. swinGather calls swinIdxTensor on every invocation (:820), which allocates a tensor and fills it with per-element SetF64 (:827-829). At batch 8 that is 16 allocations plus 16*N SetF64 calls per block, for an index array that is a pure function of (h, w, m, shift, dtype) and constant for the whole training run. swinBuildMasks (:402) likewise rebuilds numWin [n,n] mask tensors on every call.

FIX: memoize on the SwinBlock receiver, keyed by (h, w, batch, dtype), built lazily on first use: the batch-expanded forward and inverse index tensors plus the mask set. Then replace both loops with a single swinGather over the full packed tensor. swinExec1 and swinGather signatures do not change. Entirely vision-local.

VALIDATION GATE (benchmark only): extend BenchmarkSwinForwardBatched (vision/swin_batched_test.go:76) with a b.Run sweep over batch {1, 8, 32}, per backend the way internal/benchcompare/vision_train_test.go:91-105 loops gptBackends. The discriminator is the SLOPE against batch size, not the absolute number: the defect is linear in B, the fix is flat. Batch 1 is the control and must not regress. Run CPU-ref and Metal as separate rows so the SetF64 cost (paid on both) separates from the dispatch cost (GPU only).

EXPECTED: Metal 1.4-2x on top of the fused-attention fix, medium confidence; CPU 1.2-1.5x, mostly from (b), medium confidence.

BIT-IDENTITY BAR: bit-identical in both directions, and this must be asserted rather than assumed. These are pure integer permutations applied as row copies, so no value and no reduction changes. Backward is OpEmbedBackward over a bijection (permutation indices are distinct), so the atomic scatter-add has no collisions and no order sensitivity. The existing rel<1e-9 parity assertion at vision/swin_batched_test.go:66 must keep passing UNCHANGED. If it does not, stop: the fix is wrong, not the threshold.

PERFSCAN RULE REQUIRED (generalizable, part of the deliverable): add a detector for loop-invariant tensor construction inside a loop, where an allocation filled via SetF64 has no data-flow dependency on the enclosing loop variable, deriving only from receiver fields and loop-invariant locals. Add it to internal/perfscan with a positive and a negative fixture in perfscan_test.go per the existing class pattern.
