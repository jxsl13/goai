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

## T-01KYJPMP87FJBSEAATV9Z0M6YM Fuse Swin windowed attention into OpMHA/OpMHAMasked
kind: task
state: draft
created: 2026-07-27

DEPENDS ON the Swin permutation-gather task landing first — same function, and that one is provably bit-identical while this one is not.

SITE: vision/swin.go:616 (*SwinBlock).windowedAttention — window loop at :648, head loop at :662, body :663-704.

WHY HOT: SwinTransformer.Forward (swin.go:286) -> SwinBlock.forwardBatched (:366) -> windowedAttention (:404), once per block per forward and again through the tape on backward. forwardBatched passes numWin = batch*numWin (:404), so the trip count scales with batch x windows x heads.

DEFECT: each (window, head) iteration issues about 10 separate backend ops to compute one softmax(QK^T/sqrt(d))V — three OpSlice column cuts (:663-674), OpTranspose (:675), OpMatMul (:679), OpMul scale (:683), OpAdd rel-bias (:687), OpAdd shift mask (:694), OpSoftmax (:698), OpMatMul (:702) — plus three row slices and a concat per window (:649-657, :706). The backend already exposes fused OpMHA (backend/op.go:58) and OpMHAMasked (backend/op.go:60), and backend/ref/mha_masked.go:48 accepts a [heads, sq, sk] per-head additive mask, which is precisely Swin's rel-position-bias shape (swin.go:97). At the pinned BenchmarkSwinForwardBatched config (swin_batched_test.go:78) that is roughly 3,300 of about 3,900 forward dispatches.

FIX (vision-local, no nlp change): delete the head loop. Per window emit one fused call — OpMHA with AttnAttrs{Heads: b.Heads} when b.Rel is nil and masks is nil, otherwise build a [heads, n, n] additive mask once per block (rel-bias head slices from headBias at swin.go:719, stacked; shift mask broadcast-added across the head axis) and emit one OpMHAMasked. Per window: 3 row slices plus 1 fused op, down from about 35. Trainability of Rel.Table survives because OpMHAMaskedBackward gradients the additive mask (backend/ref/mha_masked.go:17).

HOST CAVEAT, do not skip: OpMHAMasked has NO Metal kernel — it runs the ref CPU kernel. The win is dispatch elimination, not GPU math. For Swin's tiny windows (n=16, dk=32) the ref kernel is microseconds against roughly 30 x 0.27ms of dispatch floor, so it still wins comfortably, but the benchmark MUST separate the Rel==nil case (GPU OpMHA) from the Rel!=nil case (CPU OpMHAMasked) or the two effects will be conflated.

VALIDATION GATE (benchmark only): BenchmarkSwinForwardBatched (vision/swin_batched_test.go:76) only contrasts per-image against batched today. Add BenchmarkSwinWindowedAttention looping backends as internal/benchcompare/vision_train_test.go:91-105 does, with sub-benchmarks relbias=on and relbias=off via vision.WithSwinRelativeBias, plus a train-step variant under autograd.NewTapeOn mirroring vision_train_test.go:113. Report img/s and a dispatch count.

EXPECTED: Metal 4-8x on Swin forward and higher on the train step, high confidence because it is arithmetic on the dispatch floor rather than a guess; CPU-ref 1.5-3x from removing dispatch overhead and about 30 intermediate allocations per window, medium confidence.

BIT-IDENTITY BAR: NOT bit-identical, tolerance only. A fused SDPA kernel changes the reduction grouping versus MatMul -> Mul -> Add -> Softmax -> MatMul, and OpDot/OpMatMul accumulate in f64 (backend/op.go:38,43) with different tiling. The rel<1e-9 assertion at swin_batched_test.go:66 will need relaxing to about 1e-6 for f32 — that relaxation is part of the change and must be justified, not quietly applied. What must stay EXACTLY equal: the window partition and shift permutation (integer index maths), the mask -Inf exclusion set, and the output shape.

PERFSCAN RULE REQUIRED: new class, hand-rolled attention where a fused op exists. AST detector: inside a for body, find the op sequence {OpTranspose, OpMatMul, OpMul or OpDiv by scalar, OpSoftmax, OpMatMul} where the second OpMatMul's operands are the OpSoftmax result and a slice of the same tensor that fed the first OpMatMul's transposed operand. Match on the backend.Op constant arguments to backend.Execute. Report only when a fused equivalent (OpMHA, OpMHAMasked, OpFlashAttn) is registered for the target backend.

## T-01KYJPMPRWF53BNYDEK5RV7KTN Remove the residual per-image dispatch loops in ViT and Mixer Forward
kind: task
state: draft
created: 2026-07-27

CONTEXT: the main batching fix (T908) landed and is verified — vision/vit.go:254 runs a packed [B·(N+1), D] path. These are the loops that SURVIVED it. Note also that internal/benchcompare/vision_train_test.go:17-23 and BENCHMARKS.md:528 still describe the pre-fix state and should be corrected alongside this work.

SITES: vision/vit.go:269-284 (patchify loop), :294-310 (class-token plus pos-embed loop), :326-333 (class-row gather). Mirrored at vision/mlpmixer.go:330-343 and :363-375, and vision/swin.go:300-314 and :342-354.

WHY HOT: ViT.Forward is the top-level entry benchmarked by internal/benchcompare/vision_train_test.go:87 and :113 — the exact functions the torch-mps comparison row was measured against.

DEFECT: three loops that are O(batch) in dispatches inside a function whose stated purpose is to make everything one large GEMM.
- :271-280 issues OpSlice plus OpReshape per image purely to hand patchify a [C,H,W] view, but patchify (vit.go:206) immediately calls Contiguous() and walks flat host memory through makeReader (vit.go:243). The two dispatches buy nothing that a base offset b*C*H*W would not.
- :296-309 issues OpSlice plus OpConcat plus OpAdd per image (3B dispatches) to prepend a shared [1,D] class token and add a shared [S,D] position table.
- :327-331 issues OpSlice per image to pull row b*S, then concatenates — a row gather written as unit slices.
At batch 8 that is roughly 51 tiny dispatches, growing linearly with batch, which is exactly the axis batching was meant to flatten.

FIX (vision-local, no nlp change):
- Patchify loop: drop OpSlice/OpReshape, extend patchify to take a base element offset and write directly into one preallocated [B·N, C·p·p] tensor. Removes 2B dispatches, B intermediates, and the B-way concat at :285.
- Class/pos loop: OpConcat([Class, emb]) once, then ONE OpEmbed with a cached [B·S] index array (row b*S -> 0, row b*S+1+j -> 1 + b*N + j), then one OpAdd against a Pos table tiled by a second cached OpEmbed. Four dispatches replace 3B+1.
- Class-row gather: one OpEmbed with index [0, S, 2S, ...] replaces B slices plus a concat.
Net at batch 8: about 51 dispatches down to about 5.
DO NOT attempt the MLPMixer per-image transpose loop (mlpmixer.go:413-421, :442-450) — a transpose permutes elements rather than rows, so the gather rewrite does not apply; a flattened-view gather would need a B*P*C-element index and is not worth the complexity.

VALIDATION GATE (benchmark only): BenchmarkViTForward and BenchmarkViTTrainStep (internal/benchcompare/vision_train_test.go:87, :113) already cover this and already loop gptBackends, so CPU-ref and Metal separate cleanly. Extend visBatch (pinned at vision_train_test.go:40) into a b.Run sweep over {1, 8, 32}. The signature of a successful fix is that per-image cost STOPS RISING with batch; batch 1 is the control and must be unchanged.

EXPECTED: Metal 1.3-1.8x at batch 8, approaching 2.5x at batch 32, medium-high confidence; CPU 1.05-1.15x, high confidence that it is small because these are metadata ops with no dispatch floor to pay.

BIT-IDENTITY BAR: bit-identical forward and backward for the computation itself — every replacement is a row copy or an elementwise OpAdd, no reduction shape changes, and the patchify rewrite reads the same source elements in the same order into the same destination offsets. EXCEPTION: Pos and Class gradients become OpEmbedBackward scatter-adds where indices repeat across images (every image touches Class row 0 and all of Pos), so on Metal the float-atomic path makes those two parameter gradients order-nondeterministic — tolerance-equal only for m.Class and m.Pos gradients, bit-identical for everything else. The forward-logits parity assertion at vision/vit_batched_test.go:78 is unaffected and must keep passing.

PERFSCAN RULE REQUIRED: new class, batch-dimension slice loop. AST detector: a for whose bound is a variable bound to x.Shape()[0] (or a parameter derived as batch), containing a backend.Execute with SliceAttrs{Axis: 0, Start: S, End: E} where (S, E) matches (b, b+1) or (b*K, (b+1)*K) for loop variable b, and whose per-iteration results are collected and OpConcat-ed on axis 0. Severity should scale with the number of backend.Execute calls in the body. Distinct from PS1003, which keys on a one-element slice LITERAL at a call site; this keys on the loop INDUCTION VARIABLE appearing as a slice bound on the leading axis with a concat sink. Also distinct from the unit-slice row gather class, whose stride is exactly one row and whose fix is OpEmbed; here the stride is a whole sub-sequence and the fix is restructuring the surrounding computation. Run the finished detector across the repo and report every site it flags.

## T-01KYQH929EEFGVGYGEE5B836KP Fuse Swin's per-window slicing — 30k allocations per batched forward
kind: task
state: draft
created: 2026-07-29

GOAL: cut the per-window dispatch overhead in Swin's windowed attention (vision/swin.go,
PS4011). This is the largest allocation count left in vision.

HOTNESS IS PROVEN, not assumed (PROC-TASK-HOTNESS-001). Panic-probing vision/swin.go:653 and
:301 under BenchmarkSwinForwardBatched/batched fires at both. The benchmark to use is
BenchmarkSwinForwardBatched, which reports img/s and both a per-image and a batched arm.

BASELINE (M2 Pro darwin/arm64, -benchtime 2x):
  SwinForwardBatched/perimage  39.5ms  202.8 img/s  26.3MB  39,352 allocs
  SwinForwardBatched/batched   18.9ms  424.0 img/s  31.0MB  30,477 allocs
For comparison ViT batched is 3,679 allocs and MLPMixer batched 2,436 — Swin is an order of
magnitude above both, and the window loop is why.

THE SHAPE: the attention body loops `for w := range numWin` and dispatches OpSlice per window
for q, k and v, then more ops inside. At 224x224 with window 7, stage 1 alone has 64 windows,
multiplied by batch and by stage. Every slice materializes a tensor.

APPROACH, and note this is a Titans-sized change rather than a one-liner — scope it
deliberately:
- The windows partition the token axis with no overlap, so the per-window slices are
  contiguous spans of one row-major buffer. A fused path can address them by offset instead
  of materializing a tensor per window.
- Follow ADR-01KYQ9PHNPEFC's precedent: keep the MATMULS on the backend, where their rounding
  is already correct, and fuse only the slicing and the elementwise chain. Attempting a fully
  fused path is what cost three iterations on Titans.
- Guard on ctx.Recorder == nil and F64 contiguity, and NAME the covered case in the guard so
  PS6003 can still see what is uncovered.

BIT-EXACTNESS: every product in the fused elementwise part must be rounded explicitly,
INCLUDING any assigned to a named local — the compiler still inlines and contracts those
(NUM-FUSED-PATH-FMA-001, and PS6012 will flag a function that pins some and not others). A
parity test in the shape of nn/gla_fused_parity_test.go — fused versus dispatch on raw float64
bits — is the gate, and it must pass before any number is reported.

VERIFY:
- go test ./vision/ -run Swin -count 1, plus -race.
- BenchmarkSwinForwardBatched interleaved, min of 3 runs per arm, reporting allocs/B beside
  ns/op. The allocation count is the headline here; wall-clock may move less.
- Report BOTH arms: if the per-image arm improves and the batched one does not, say so.

SCOPE: vision/swin.go only. vit.go and mlpmixer.go have the same per-batch-element shape at
270/295 and 330/364/413/442 but an order of magnitude fewer allocations, so they are not worth
the same effort until this one is measured.
