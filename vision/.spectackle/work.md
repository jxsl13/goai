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

## T-01KYJPMNTGFPXRXQ846DZDSFYK Replace the MAE unit-slice row gathers with OpEmbed
kind: task
state: draft
created: 2026-07-27

SITE: vision/mae.go:450 gatherRows and vision/mae.go:428 unshuffleRows.

WHY HOT: four call sites per training step, all on the hot path — Encode (mae.go:489), Reconstruct (mae.go:537), and maskedMSE twice (mae.go:572, :576). MAELoss (mae.go:598) is the documented training objective, so this runs every step, forward and again through the tape.

DEFECT: gatherRows issues one backend.OpSlice{Axis:0, Start:i, End:i+1} PER ROW (mae.go:453-454) and concatenates; unshuffleRows does the same (mae.go:431-432). The backend already has OpEmbed, documented at backend/op.go:63 as exactly a row gather: (table[n,d], idx[m]) -> out[m,d], with a gradient that scatter-adds to the table. The same package already uses it correctly — swinGather (vision/swin.go:819-821) is swinExec1(ctx, backend.OpEmbed, nil, t, swinIdxTensor(...)). MAE simply did not get the same treatment. At MAE defaults (mae.go:203-206: patch 4, maskRatio 0.75) on a 32x32 image, S=64, keep=16, masked=48, that is 128 unit slices plus 4 concats per forward against roughly 50 dispatches for the actual encoder and decoder blocks — about 72% of forward dispatches doing index bookkeeping. At ViT-Base MAE geometry (S=196, keep=49, masked=147) it is 392 unit slices.

FIX (vision-local, mechanical, about 15 lines): rewrite gatherRows as a single OpEmbed using the existing helper pattern at swin.go:825 (swinIdxTensor builds the rank-1 float index tensor OpEmbed expects). For unshuffleRows, concatenate [proj ; maskToken] once, then one OpEmbed whose index array maps each original position to either its proj row or to the single mask-token row — one concat plus one gather replaces keepCount slices plus an S-way concat.

VALIDATION GATE (benchmark only): NO MAE benchmark exists anywhere in vision/. Write vision/mae_bench_test.go with BenchmarkMAELoss and BenchmarkMAEReconstruct, looping backends as internal/benchcompare/vision_train_test.go:91-105 does so CPU-ref and Metal are separate rows, and parameterize WithMAEMaskRatio over {0.5, 0.75, 0.9}. The cost of the defect is linear in S, so the sweep is what proves the win is the gather and not the encoder.

EXPECTED: Metal 3-5x on MAELoss at S=64, growing with S, high confidence. Note OpEmbed on Metal is a deliberate host-side f32 copy of roughly 50us (see the embedF32 comment at backend/metal/metal.go:1049-1052), so this is dispatch elimination, not GPU math. CPU 1.3-2x, medium confidence.

BIT-IDENTITY BAR, split — read carefully before touching tests:
- Forward: BIT-IDENTICAL. OpEmbed is a pure row copy, as is slice+concat. Same bytes.
- gatherRows backward: BIT-IDENTICAL. keep and masked are disjoint (mae.go:373-379), so OpEmbedBackward's scatter-add hits each row exactly once.
- unshuffleRows backward: TOLERANCE ONLY. m.MaskToken is placed at every masked slot (mae.go:439), so its gradient sums over 48-147 rows. embedBackwardF32 on Metal uses float atomics (backend/metal/metal.go:1001-1002), which is order-nondeterministic, so the mask-token gradient differs run to run at f32 epsilon. Any gradcheck in mae_test.go asserting exact equality on MaskToken must move to a documented tolerance, and that relaxation must be justified in the commit rather than done silently.

PERFSCAN RULE REQUIRED: new class, unit-slice row gather. AST detector: a for loop whose body calls backend.Execute with backend.OpSlice and SliceAttrs{Axis: 0, Start: E, End: E2} where E2 is syntactically E+1 (or Start: i, End: i+1 with i the loop variable), and whose results are collected into a slice later passed to OpConcat on the same axis. Structurally DISTINCT from PS1003: PS1003 fires on a single-element slice literal handed to an already-batch-capable API (right op, wrong call shape); here there is no slice literal, the loop index IS the slice bound, the op itself is the wrong primitive, and the remedy is a different op (OpEmbed). State that distinction in the rule doc so the two do not get merged later.
