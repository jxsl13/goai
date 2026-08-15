---
schema: v1
---

## T-01KYJNDQJZEW0SMH58S84Z23PM Add an SGLang datapoint beside vLLM in the serving comparison
kind: task
state: approved
created: 2026-07-27
targets: docs/benchmarking.md

Extend the serving benchmark from three-way to four-way. SGLang was confirmed to install on the worker box but was never measured.

Scope: worker-machine only (RTX 3060, TinyLlama-1.1B). Use the identical protocol as the existing vLLM rows: batch=1 decode tg128, prefill pp128, n=16 and n=64 aggregates, sequential GPU-exclusive runs, versions recorded.

Output: a three-way section row in docs/benchmarking.md and a row in BENCHMARKS.md section 1.

Value: SGLang's RadixAttention prefix reuse is the industry mirror of this repo's prefix-caching arc, so a direct comparison sharpens that story. Precision must be matched and disclosed on both sides.

Migrated from cavekit SPEC.md T888.

## T-01KYJNDR38E4ZSN52KVM0PC5J9 Extend the trained-optimizer comparison with APOLLO and Q-GaLore
kind: task
state: approved
created: 2026-07-27
targets: docs/training.md

The nn package ships APOLLO and Q-GaLore, but the trained-optimizer zoo in llamagpu/optimizers_trained_test.go and the tables in docs/training.md still end at the earlier optimizer set. Godoc alone cannot give the comparative same-task cross-entropy-after-120-steps measurement, which is the value here.

Work: add APOLLO (seed-only projection default) and Q-GaLore (default QuantBits, plus the QuantBits=0 GaLore-collapse datapoint) rows to the zoo and wrapper harness using the same task, step count and protocol; refresh the zoo and wrapper tables in docs/training.md; record which defaults were used, with the research grounding for each.

Also fix the stale section-range header in training.md so it names the range actually covered.

Migrated from cavekit SPEC.md T891.

## T-01KYJNDT44EKJAXN8W0Y4QFZCE Batch the ViT encoder instead of looping over the batch dimension
kind: task
state: done
created: 2026-07-27
targets: vision

vision.ViT.Forward processes a [batch,C,H,W] input by looping over the batch: slice image b, forwardOne, concat. A batch of N therefore runs as N independent length-(patches+1) encoder forward and backward passes. On GPU each per-image op pays the roughly 0.27ms Metal/Vulkan dispatch floor, multiplied by N, which measured about 40x behind torch-mps for ViT training. On CPU the same defect costs only 2.6x to 4.2x because there is no dispatch floor, so this is a GPU lever specifically.

Fix: carry a leading batch dimension through patch-embed, class-token prepend, positional embedding, the MHA blocks, final layer norm and the head, so attention runs one [N, patches+1, D] pass.

Feasibility, already scoped: nlp.MHA.Forward asserts x.Ndim()==2, so the shared attention is 2D-only and the ViT loops per image because attention cannot batch. Three routes: (a) add a batch dimension to OpMHA, nlp.MHA and every backend MHA kernel, the real fix but deep and cross-cutting into the worker's attention kernels; (b) a ViT-local batched attention in vision/ that bypasses nlp.MHA, moderate but duplicates attention; (c) wire the batched recorder to the autograd tape, currently parked.

Correctness bar: bit-identical to the per-image loop at batch=1 and equal to looping for batch>1, pinned by a gradient check and a per-image-versus-batched parity test.

Benchmark harness already exists at internal/benchcompare/vision_train_test.go.

Migrated from cavekit SPEC.md T908.

## T-01KYJNDTK2E8A9BK9SGAZ4VKYS Hoist per-layer Attrs boxing across the remaining decode models
kind: task
state: approved
created: 2026-07-27
targets: nlp

Class-audit sweep in nlp. A per-layer decode closure builds a backend.RoPEAttrs or backend.AttnAttrs struct literal and boxes it into the Attrs interface inside the loop, although the fields are layer-invariant.

Apply the same mechanical, provably bit-identical hoist already done for one model to the remaining decode models: cohere, falcon, gemma, gemma2, cla, blt and their siblings. Find the sites by searching nlp/*decode*.go for a dispatch call passing a backend.RoPEAttrs or backend.AttnAttrs literal.

Measure per model or as a batch, with a same-session A/B and bit-identity verification before shipping.

Coordination: re-check for a scope collision per file before editing, as a parallel worker is active in the mamba2 and quantized-mamba2 files.

Migrated from cavekit SPEC.md T956.

## ADR-01KYJNF428F8Q9RQTABB1ZSVPC What is the agent commit and push authority on this repo?
kind: adr
state: submitted
created: 2026-07-27
context: The migrated cavekit constraint C8 says the loop must never commit or push without explicit user permission and works in the working tree only. The repository history contradicts this: work lands as autonomous branch-plus-pull-request pushes, and the worker runner protocol RUN2 mandates PR-only pushes with manual merge after green CI. C8 was therefore not migrated as a rule, because encoding either reading without confirmation would put a false contract in the spec. The spec server itself runs with git.mode=offline, so it commits but never pushes.
status: proposed

kind: radio
option: Autonomous branch + PR push, manual merge after green CI (matches current practice)
option: Autonomous commit and push including direct to main
option: Working tree only - explicit user permission required before any commit or push

## R-01KYJWDW8CESXSAZNJ0694FGJJ PS1001 narrowing: configured shapeMethods measured, rejected
kind: research
state: draft
created: 2026-07-27

NEGATIVE RESULT - narrowing option (a) is dead; do not retry it.

Implemented the recorded next step: a configured shapeMethods list in perfscan.json (value ["Shape"]), gating the numelIdents widening on `x := <configuredShapeMethod>()[i]` instead of on any indexed call. Measured tree-wide with -config: PS1001 went 44 -> 266, i.e. IDENTICAL to the unnarrowed any-indexed-call widening (573 total candidates vs the 349 baseline; PS2001 also picked up 2, since it consumes the same numelIdents set). Configuring the method name therefore narrows nothing: t.Shape()[k] IS the dominant bound form in this tree, so gating on it selects the same population as gating on nothing.

Second, independent problem in the same run: the per-site greps reported zero findings tree-wide while the run's own summary line reported 266. The two contradict, so nothing about WHICH sites the variant reports was established - including whether it reports the nlp/kvevict.go target at all. Only the aggregate 266 is trustworthy. Any future variant must be measured with the summary line PLUS a per-site check verified consistent with it before its site-level behavior is believed.

Baseline restored (git checkout of perfscan.go + perfscan.json); re-verified at PS1001=44 / 349 total, tree clean.

Confirmed target shape, read back from the reverted tree - nlp/kvevict.go GatherRows:
    d := t.Shape()[1]
    out := tensor.New(t.Dtype(), tensor.Shape{len(idx), d})
    for r, src := range idx { for j := range d { out.SetF64(t.AtF64(src, j), r, j) } }

What this leaves: the remaining narrowing axis is NOT the callee name but the USE of the bound ident. Require the ident to bound a loop whose body holds a per-element accessor indexed by BOTH that loop var and an outer one (the r,j pair above) - narrow on the loop-nest shape, not on where the dimension came from. Unmeasured. Measure the tree-wide count first; only look at sites once the count is small enough to triage by hand.

Standing caution: every intermediate conclusion in this investigation has been wrong at least once, including two measurement-apparatus failures (perfscan run without -config, and now contradictory grep-vs-summary counts). Trust only numbers re-derived on a clean tree.

## T-01KYJWGBGRER59EX4MPX6S8VE7 cpu broadcast odometer: hoist innermost axis into runs (PS4004 triage)
kind: task
state: draft
created: 2026-07-27

PS4004 triage over all 22 findings, by hotness. One real candidate, one detector false-positive class.

CANDIDATE (highest leverage): backend/cpu/elementwise.go:50 (F32) and :65 (F64) - the broadcast-materialize odometer. Structure:
    for pos := 0; pos < n; pos++ { dst[pos] = src[off]; <per-element odometer over ndo axes> }
This is the SAME shape already optimized in backend/ref/broadcast.go, where hoisting the innermost axis out of the odometer into runs measured 4.49x interleaved (broadcastRuns[T refFloat], with a rank-0 guard). The innermost effective stride bstride[ndo-1] takes exactly two values in practice: 0 (that axis is broadcast -> the whole run is one repeated value, a fill) and 1 (contiguous -> one copy()). Same switch sInner {0,1,default} split applies verbatim; the default arm keeps the strided walk so the transform stays total.

Leverage: this is backend/cpu, the production elementwise path whenever shapes need broadcasting, so it is hit far more often than the ref twin. dst is written in ascending pos order and src is read at the identical offsets, so it is a pure memmove reordering with no arithmetic - bit-identity is by construction (PS4004's own argument), not by rounding analysis.

VALIDATION REQUIRED BEFORE IMPLEMENTING (main agent, benchmarks only): interleaved A/B via file-copy toggle, same host and session, medians, PLUS an unaffected control benchmark that must not move. The earlier broadcast work is precedent for why: a non-interleaved run reported 5.5x while the untouched control drifted 46.4 -> 37.0 us with no code change; interleaved re-measurement gave the true 4.49x. Bit-identity proven by byte-comparing raw output bits across the toggle, and the equivalence test must be non-vacuous (mutate the new path on purpose, confirm red) across shape regimes covering: innermost broadcast (stride 0), innermost contiguous (stride 1), strided default arm, and rank 0/1 edge cases, for both F32 and F64 arms.

FALSE-POSITIVE CLASS found in the same triage: backend/cpu/gemm.go:68 already sits in the ELSE branch of an `if s1 == 1 { copy(drow, src[so:so+cols]) }` guard - the contiguous run is handled by copy() and the flagged loop is the strided fallback that must stay. PS4004 cannot see this because it inspects the loop, not its enclosing branch. Detector refinement (generalizable, AST-only, no go/types): suppress PS4004 when a SIBLING branch of the enclosing if/else already contains a copy() whose arguments name the same dst and src identifiers as the flagged assignment. Expect this to also clear other guarded fallbacks in the list. Measure the finding-count delta against the 22 baseline and confirm the surviving elementwise.go sites still report.

Remaining 19 findings are index/shape bookkeeping over tiny slices (tensor/strides.go:76, backend/einsum.go:101 and :128, nlp/spm.go:243, nlp/sample.go:677 and :895) or one-shot weight-load paths (nlp/gpt2_hf.go x3, nlp/llama_gguf.go:465, nlp/quant_mixtral_gguf.go:370, llamagpu/cuda.go:186, backend/cuda/cuda_quant_iq3.go:199) - not hot per token, no benchmark would resolve them above noise. Not worth a task; left as standing advisory findings.

## R-01KYJWN7RTF1E8VP2F8APHC8HD cpu broadcastRuns not validated: benchmarks never enter broadcastContig
kind: research
state: draft
created: 2026-07-27

NOT VALIDATED - benchmark attempt was vacuous, and it corrects a leverage claim in this task's original body.

CORRECTION to the original leverage estimate. The original body claimed broadcastContig is "the production elementwise path whenever shapes need broadcasting, hit far more often than the ref twin". That is wrong. broadcastContig has exactly ONE call site (backend/cpu/elementwise.go:241) and it sits BEHIND a trailing-block fast path: when the smaller operand is a contiguous trailing block of the output shape (the [n] bias over [m,n] case, and every shape like it), the kernel dispatches to bcastBlockApply, which applies the op directly against the repeating block and never materializes anything. broadcastContig is only reached for GENERAL broadcasting that the block path does not cover - broadcast on a non-trailing axis, or both operands broadcasting. Real leverage is the residual after that fast path, far smaller than stated.

MEASUREMENT PERFORMED AND WHY IT PROVES NOTHING. Implemented the runs transform (generic broadcastRuns[T float32|float64], switch sInner {0 fill, 1 copy, default strided}, rank-0 guard, both dtype arms) - builds clean, gofmt clean, full backend/cpu test package green in 13.5s. Interleaved A/B by file-copy toggle, three A-B alternations, -benchtime=200x -count=3 per round, medians over 9 samples per variant:
  AddBiasF32_128x2048: A 98290 ns, B 100667 ns
  AddBiasF32_8x32000:  A 136370 ns, B 145240 ns
  control AddF64_4K:   A 6673 ns,  B 6231 ns
The control moved 6.6 percent with no code change touching it, so the noise floor swamps both deltas. More decisively: all three benchmarks route through bcastBlockApply or the same-shape SIMD path, so NONE of them execute broadcastContig. The numbers measure unchanged code. They are not evidence against the transform; they are not evidence at all.

WHAT A REAL VALIDATION NEEDS. A new benchmark that actually enters broadcastContig - construct operands whose broadcast is not a trailing contiguous block, for example [m,1,k] against [m,j,k] (middle-axis broadcast, innermost stride 1 so the copy arm runs) and [m,n] against [m,1] style shapes that put stride 0 innermost so the fill arm runs. Assert entry into the path before trusting any delta (a temporary counter or a deliberate panic in broadcastContig confirms the benchmark reaches it - the failure mode above is precisely that this was assumed rather than checked). Only then run the interleaved A/B with the control.

STATUS: reverted, tree clean at baseline. The transform itself is sound and test-green; it is the leverage and the measurement surface that are unestablished. Do not implement until a benchmark proven to enter the path shows a win outside the noise floor.

METHOD LESSON worth carrying: confirm the benchmark reaches the code under test BEFORE reading its numbers. An existing benchmark whose name matches the feature is not proof it exercises the branch, especially where a fast path shadows the general one.

## R-01KYJWXFVZF42RYNW92Y1T66EN cpu general-broadcast runs transform validated and shipped (5.29x)
kind: research
state: draft
created: 2026-07-27

VALIDATED AND SHIPPED - commit 5bfa77b. Supersedes the not-validated finding; the earlier attempt failed on measurement surface, not on the transform.

WHAT SHIPPED: broadcastRuns[T float32|float64] in backend/cpu/elementwise.go hoists the innermost axis out of the general-broadcast odometer. Run switch: stride 0 fill, stride 1 copy, default strided walk (total). Rank-0 guard for scalar output. Both dtype arms route through it.

MEASURED, interleaved A/B by file-copy toggle, same host/session, medians of 9 samples, -benchtime=300x -count=3, three A-B alternations:
  BroadcastMidAxisF32_32x64x256  1303174 -> 246575 ns  5.29x
  BroadcastMidAxisF64_32x64x256  1380355 -> 366238 ns  3.77x
  BroadcastInnerF32_32x64x256     224228 -> 196446 ns  1.14x
  control AddF64_4K                  3802 ->   3725 ns  1.02x
Control drift 2 percent, so the mid-axis deltas are far outside noise.

THE DECISIVE METHOD STEP: the benchmarks were proven to enter broadcastContig by temporarily panicking inside it BEFORE any number was read. The prior round's AddBias benchmarks name-matched the feature but never reached the code, because bcastBlockApply shadows the general path for trailing-block shapes; those numbers measured unchanged code. New benchmarks use middle-axis shapes ([32,1,256] against [32,64,256]) that the block fast path does not cover.

BIT-IDENTITY: new TestGeneralBroadcastBitIdenticalToRef compares cpu against ref on RAW BITS (math.Float64bits, no tolerance) across six shape regimes x F32/F64.

NON-VACUITY, established by mutation rather than assumed - and this caught a real overclaim. Corrupting the contiguous-run arm turns the new test red. Corrupting the FILL arm did NOT: the case named for stride 0 does not actually reach that arm. A separate panic probe showed the fill arm IS reached elsewhere in the package suite, and a full-suite mutation run confirmed the suite catches a fill-arm defect. The test comment now states exactly which arm it covers and warns that case names are not proof of which arm ran.

RULE COVERAGE (standing requirement): the pattern is already machine-findable - PS4004 is what surfaced this site. No new rule needed for this optimization. The separately identified PS4004 refinement (suppress when a sibling branch already holds a copy() over the same dst/src, which would clear the backend/cpu/gemm.go:68 false positive) remains open and unimplemented.

LEVERAGE NOTE CARRIED FORWARD: this path is the residual after bcastBlockApply, not every broadcasting elementwise op. The 5.29x is real but applies to general broadcasts (non-trailing-block), not to the bias-add shapes that dominate transformer inference.

METHOD LESSON, now twice-earned: confirm the benchmark reaches the code under test, and confirm the test reaches the branch it claims, both by deliberate breakage. Name matching proves neither.

## T-01KYJYPQ38E938WYV6HGX9BMF5 PS5001 precision: suppress integer divides via the modulo-sibling proof
kind: task
state: draft
created: 2026-07-27

PS5001 (97 findings, the largest class) conflates float divides with INTEGER index decomposition, where its advice is not merely non-bit-identical but arithmetically wrong. Fix the precision before anyone triages the class.

EVIDENCE, sampled from backend:
  backend/cpu/conv.go:159   ni, rem := r/hw, r%hw          <- integer
  backend/cpu/conv.go:205   ni := r / (ho * wo)            <- integer (paired with r % (ho*wo) on the next line)
  backend/cpu/conv_backward.go:107, :225, gemm_amx_arm64.go:138, gemm_avx2_amd64.go:67 - same index-decomposition shape, divisors named wo, nb, sgCols
  backend/cpu/attn_extra.go:171  or[d] = T(acc[d] / l)     <- genuine float divide, a real candidate
For an integer divide, the recommended `inv := 1/hw; x * inv` evaluates 1/hw as integer zero. Following the advice silently zeroes the result. The rule's message already hedges with "verify float + intent", which is an admission that the detector cannot tell - it is AST-only with no go/types, by design.

SOUND TYPE-FREE DISCRIMINATOR: Go forbids % on floating-point operands, so the presence of `a % b` PROVES a and b are integers. Suppress a PS5001 finding when the same numerator/denominator expression pair also appears under token.REM within the enclosing function. This is a proof, not a heuristic - it can never suppress a genuine float divide, because a float divide cannot have a modulo sibling. It catches the dominant idiom directly: index decomposition is nearly always written as the /,% pair, whether on the same line (r/hw, r%hw) or adjacent lines with a parenthesized divisor (r/(ho*wo), r%(ho*wo)). Compare expressions structurally (a small AST-equality helper over Ident/BasicLit/BinaryExpr/ParenExpr), not by printed text, so parenthesization does not defeat the match.

IMPLEMENTATION NOTES: internal/perfscan/perfscan.go, alongside the existing PS5001 detector. AST-only, no configuration - keep it a language-shape check like its siblings. Precedent for this shape of fix: the PS4004 sibling-branch suppression (commit c1b4f74) and PS4005's startsBelowInnermost discriminator (0c671c2), both of which cut a false-positive class without touching true positives.

VERIFICATION - count delta is NOT sufficient on its own, verify the SET:
1. Capture the 97-finding baseline as a sorted file:line:col set before changing anything.
2. After the change, diff the sets both ways. Every disappearance must be a divide with a modulo sibling - inspect each one, do not sample. Nothing may appear.
3. attn_extra.go:171 MUST survive; it is the known true positive and is the floor against over-suppression.
4. Add a unit test in the perfscan package covering both a /,% pair (suppressed) and a bare float divide (still reported), so the discriminator cannot silently regress.

RESIDUAL, state it in the commit rather than implying the class is clean: divisors with no modulo sibling (sgCols, nb) may still be integer and stay unresolved by this fix. This raises precision; it does not make the class safe to apply blindly. The remaining findings still need per-site float/intent confirmation, and the standing constraint holds - reciprocal multiplication is NOT bit-identical, so it is admissible only for a continuous output (gradient, moment, probability) and never where the value feeds round, quantize, argmax or any comparison against a threshold.

## ADR-01KYJYY74VE27BSEH9VGZSNFMK PS5001 reciprocal-multiply trades bit-identity for speed - accept, reject, or case-by-case
kind: adr
state: draft
created: 2026-07-27

DECISION NEEDED BEFORE ANY PS5001 SITE IS IMPLEMENTED: reciprocal-multiply is NOT bit-identical, which puts the whole class in tension with this repo's standing bit-identity discipline. Every optimization shipped in this line of work so far (ref broadcast 4.49x, cpu broadcast 5.29x, tensor gather 3.14x, ref argmax 1.78x) was bit-identical BY CONSTRUCTION - pure traversal reordering with no arithmetic change. PS5001 is the first class where the win requires accepting a numerics change, so it cannot be waved through on the same authority.

THE TRANSFORM: x / c becomes x * (1/c) for a loop-invariant c. The two differ by up to a half ulp per element, because 1/c is itself rounded before the multiply. There is NO bit-identical variant - the divide is by a loop-invariant, but each numerator differs, so nothing can be hoisted without changing the arithmetic. Rejecting bit-identity is the entire cost of the class; there is no third option to look for.

THE SURVIVING TRUE POSITIVE, and it is hot: backend/cpu/attn_extra.go:171,
    or := out[qBase : qBase+dk : qBase+dk]
    for d := range or { or[d] = T(acc[d] / l) }
This is attention output normalization by the softmax denominator - it runs per query, per head, per head-dimension, on every token of every decode step. Leverage is plausibly the highest of any remaining candidate. Expected gain per the rule's own historical note is roughly 1.2-1.5x on the divide itself, which is a fraction of MHA, so the end-to-end effect must be measured rather than assumed.

WHY THIS NEEDS THE USER, not an agent judgment call:
1. Attention output is a CONTINUOUS value that feeds further matmuls, which is exactly the case the rule labels admissible. But admissible-by-rule is not the same as acceptable-in-this-repo, where bit-identity has been treated as the gate for shipping.
2. Concrete blast radius: cross-path equality tests compare two GoAI code paths against each other (streaming vs non-streaming decode is one). A half-ulp change applied to only one path can break such a test even though both remain numerically correct. Which tests would move is unknown until measured.
3. The nn package currently has two RED bit-identity tests (TestGradFnBitIdenticalToSlowPath, TestEMAUpdateBitIdenticalToSlowPath, both single-ulp fast-vs-slow divergences). Verified as pre-existing - identical with and without the tensor gatherCast change - and untouched because nn is the parallel worker's lane. Loosening numerics elsewhere while ulp-level guards are already failing would make attribution of any future divergence materially harder.

OPTIONS:
(a) Reject the class outright. Keep bit-identity absolute; PS5001 becomes advisory-only and its 78 findings are never acted on. Costs the 1.2-1.5x on genuinely hot divides.
(b) Accept for continuous outputs only, with a written rule fixing the boundary: admissible for gradients, moments, probabilities and activations; forbidden where the value feeds round, quantize, argmax, or a comparison against a threshold. Each site still needs per-site confirmation that its output is continuous, plus a tolerance-based test replacing the bit-identical one, and the loosened assertion must be justified in the commit.
(c) Accept case-by-case, each site its own decision. Highest friction, but no blanket precedent.

RECOMMENDATION: (b), because the boundary is exactly what the detector already encodes and a written rule makes it reviewable, but ONLY after the two red nn tests are resolved - otherwise a deliberate ulp change lands on top of an unexplained one and neither can be attributed cleanly.

NOT MEASURED YET, deliberately: no A/B was run on the attn_extra site, because measuring first would invite shipping on the strength of a good number before the numerics question is settled. If the answer is (a), the measurement is wasted work. If (b) or (c), the mandatory sequence applies - prove entry by a temporary panic, interleaved A/B with an unaffected control, then a tolerance test with per-arm mutation - and additionally a full-tree test run to find every cross-path equality test the half-ulp shift disturbs.

## R-01KYKRP643E13B2FF1SGBJAX4T PS3002 triage: radix advice is conditional and mostly inapplicable
kind: research
state: draft
created: 2026-07-28

PS3002 (43 findings) recommends replacing a comparator sort with an LSD radix on the key bits, but the advice is CONDITIONAL in its own message - "if the key is a monotonic float/int over a large slice" - and the detector cannot check either condition. Sampling the non-autograd, non-nn sites suggests the class is dominated by sites meeting neither.

SAMPLED:
  backend/offload.go:142      SliceStable over offload candidates - slice length is the number of offloadable tensors, small.
  backend/ref/svd.go:101      SliceStable over singular values - length is a matrix dimension, small.
  format/safetensors/safetensors.go:344  Slice over tensor entries - a STRING key, where radix-on-float-bits does not apply at all, and a one-shot load path besides.
The one historically confirmed win behind this rule (top-p / typical sampling, 1.9-2.25x) has a numeric key over a vocab-sized slice, which is exactly the shape none of these three have.

THIS IS THE SAME DEFECT SHAPE AS PS5001 (fixed in 6affca8): a rule whose recommendation is sound for one operand type and inapplicable or wrong for another, with the message hedging because the AST-only detector cannot tell. PS5001 was fixable because Go supplies a type PROOF - % is illegal on floats. The question here is whether an equivalent proof exists.

CANDIDATE DISCRIMINATORS, none yet verified, listed in order of soundness:
1. String-key proof: a comparator whose body compares operands with strings.Compare, strings.Less, or indexes a field named in a string-typed context. Weak - field names are not types, so this is a heuristic, not a proof. Do NOT ship it as a suppression on that basis; at most it could downgrade the message.
2. Small-slice exclusion: a sort whose slice is a struct field or local sized by a rank/dimension rather than an element count. This mirrors PS4004's isCountedLoop distinction, which is already implemented in this file and proven workable - reuse that predicate rather than inventing a second one.
3. Reframe rather than suppress: keep all 43 findings but make the message state the two preconditions as a CHECKLIST the reader must confirm, instead of burying them in a conditional clause. Cheapest, no false-suppression risk, and honest about what an AST-only rule can know.

RECOMMENDATION: option 3 first, since it costs nothing and cannot lose a true positive, then investigate 2. Explicitly reject 1 as a suppression - a heuristic that silences findings is worse than a noisy rule, and PS4005's near-miss (where the rule flagged its own fix until startsBelowInnermost was added) is the standing reminder that suppression logic must be a proof, not a guess.

VERIFICATION when acting: capture the 43-finding set first, diff both directions, inspect every disappearance individually rather than sampling, and keep a known true positive as the floor - the top-p sampling site in nlp is the natural one. This is the procedure that caught the PS5001 suppressions cleanly.

NOT DONE: no site here was benchmarked. On present evidence none of the three sampled sites justifies one, and a measurement on a small slice would only produce noise.

## R-01KYKRV4PCENVT0XQC3BGC0VNK PS4001 triage: block-scale reads are not bulk-copyable decodes
kind: research
state: draft
created: 2026-07-28

PS4001 (30 findings) is the THIRD detector this session whose recommendation applies to a strict subset of what it reports, with the message hedging instead of the check discriminating (after PS5001 integer divides, fixed in 6affca8, and PS3002 sort preconditions, reframed in c8b8917). The pattern is now established enough to be worth naming as a class of detector defect.

WHAT IT ADVISES: replace a per-element binary.LittleEndian.Uint16 decode with a single bulk copy, valid because on a little-endian host the on-disk bytes already match the in-memory layout for verbatim-bit values.

WHY THE SAMPLED SITES DO NOT QUALIFY - format/gguf/gguf.go:612 (dequantQ8_0Into) and :633 (dequantQ4_0Into):
    blk := raw[b*34 : b*34+34]
    d := f16ToF32(binary.LittleEndian.Uint16(blk))
The Uint16 reads ONE SCALE PER BLOCK - once per 32 elements, not once per element - and its result is immediately converted by f16ToF32 rather than stored verbatim. There is no stream of u16 values to bulk copy; the bytes that dominate the loop are the quantized payload, which is decoded by genuine arithmetic. The rule's own text already exempts this ("a path that genuinely converts per element is fine"), so the sites are exempt by the rule's own terms and should never have been reported. backend/cuda/cuda_quant_q4k_pre.go:69 has the same block-scale shape.

DISCRIMINATOR, AST-only and matching the rule's stated exemption: a bulk-copyable decode stores the Uint16 result VERBATIM into a destination element - the call is the direct right-hand side of an assignment to an index expression, `dst[i] = binary.LittleEndian.Uint16(src[2*i:])`. A genuine conversion wraps the call in something else - another call (f16ToF32), a cast, or arithmetic. Suppress when the Uint16 call is not the direct RHS of an index assignment. This is precise rather than merely heuristic, because the transform being recommended is only meaningful when the value is stored unchanged; if anything is applied to it, no copy can reproduce the loop.

SECOND CONSTRAINT, unrelated to precision and not currently enforced anywhere: the transform is only correct on a little-endian host. Any site that is acted on needs a build tag or a runtime byte-order check, not an unconditional bulk copy, or it silently corrupts data on big-endian. The message should say so - it currently mentions little-endian as the justification without stating it as a requirement on the fix.

VERIFICATION when acting: capture the 30-finding set, diff both directions, inspect every disappearance individually, and keep a genuine verbatim-copy site as the floor against over-suppression - if none exists in the tree, that is itself the finding, and the rule has no true positives here.

NOT MEASURED: no benchmark was run. GGUF load is a one-shot path, though a large one (gigabytes for a big model), so load-time and peak-allocation are the metrics that would matter, not per-token throughput. Worth measuring only after a true positive is confirmed to exist.

## R-01KYKS0FGVEQ6V56X4S52NERD8 PS4001 discriminator built and reverted: survivors unverified
kind: research
state: draft
created: 2026-07-28

PS4001 verbatim-store discriminator: IMPLEMENTED, MEASURED, THEN REVERTED as unverified. Do not re-apply without completing the verification below - the code is sound in principle but its result was never confirmed.

WHAT WAS BUILT: a storesVerbatim(parent, call) predicate gating the PS4001 emission, requiring the decode call to be the direct right-hand side of an assignment into an index expression (`dst[i] = binary.LittleEndian.Uint16(src[2*i:])`). Rationale: a bulk copy can only replace a decode loop when the decoded value reaches its destination unchanged, so any wrapping call, cast or arithmetic disqualifies the site. This is the exemption PS4001's prose already granted but never enforced. The message was also rewritten to state little-endian as a REQUIREMENT ON THE FIX (build tag or runtime byte-order check) rather than merely as its justification - an unconditional bulk copy corrupts data on a big-endian host.

MEASURED: PS4001 30 -> 6 findings, nothing newly appearing. The two known false positives (format/gguf/gguf.go:612 and :633, the dequantQ8_0Into and dequantQ4_0Into block-scale reads that feed f16ToF32) were correctly silenced.

WHY IT WAS REVERTED: the six survivors could not be confirmed. Findings are emitted at loop.Pos(), and searching 20 lines forward from each surviving position (classic/gbm_hist.go:50, format/gguf/iq2s.go:141, iq2xs.go:112, iq2xxs.go:111, iq3s.go:113, iq3xxs.go:96) turned up NO binary.* or LittleEndian call at all. Either the decode sits further inside a long loop body, or binaryDecodeCall matches a form that was not searched for. Without knowing which, there is no verified floor: if storesVerbatim matches something other than intended, the six survivors may be wrong AND the 24 suppressions may include true positives. A suppression of 80 percent of a class cannot rest on an unchecked predicate.

TO FINISH: (1) find what binaryDecodeCall actually matches - read its definition and the helper names it accepts, since the survivors suggest it is broader than binary.LittleEndian.*; (2) locate the decode inside each of the six survivors and confirm each is a genuine `dst[i] = decode(...)` verbatim store; (3) inspect all 24 suppressions individually rather than trusting the predicate's definition, which would be circular; (4) only then re-apply. The patch is small enough to retype from this description; it is one predicate plus a call-site conjunct plus a message rewrite.

METHOD NOTE: the first patch attempt silently no-oped because the replacement anchor was indented with spaces while the file uses tabs, and the resulting 30 -> 30 reading was briefly mistaken for a real measurement of an unchanged detector. Any future edit here should assert its anchor matched before believing the numbers that follow.

## R-01KYKS353EFD0VD715GZP8M55N PS4001 resumed: storesVerbatim insufficient, IQ sign-trick sites survive
kind: research
state: draft
created: 2026-07-28

Resumption of the reverted PS4001 discriminator. The blocker is resolved and the conclusion CHANGES: storesVerbatim is necessary but NOT sufficient, so re-applying it alone would have been wrong.

BLOCKER RESOLVED: binaryDecodeCall matches math.Float32frombits / Float64frombits (mathBitsCallees) in addition to binary.LittleEndian.* and binary.BigEndian.*. That is why searching the survivors for "binary." found nothing and the floor could not be established last round. Any future inspection of this detector must search for the math.*frombits forms too.

THE SIX SURVIVORS, now read:
  classic/gbm_hist.go:56     col[i] = math.Float64frombits(u)                                  <- genuine shape
  format/gguf/iq2s.go:144    y[k] = math.Float32frombits(math.Float32bits(db*gridRow[k]) ^ sbit)
  format/gguf/iq2xs.go:115   same shape
  format/gguf/iq2xxs.go:116  same shape
  format/gguf/iq3s.go:117    y[k] = math.Float32frombits(math.Float32bits(db*r1[k]) ^ sb1)
  format/gguf/iq3xxs.go:100  same shape
Five of the six are STILL false positives. They are direct index assignments, so storesVerbatim admits them, but the value is not read verbatim from anywhere - it is a sign-bit XOR applied to a computed product (db * gridRow[k]). This is the IQ-quant sign trick, arithmetic wearing a bit-cast, and no bulk copy can reproduce it. Precision would have gone 30 -> 6 with 5 of 6 still wrong.

SECOND CONJUNCT REQUIRED: the decoded BITS must themselves be read, not computed. AST-only test - reject when the decode call's argument is a BinaryExpr (or contains one), since an XOR/OR/shift over a computed value is by construction not a verbatim source read. Expected to drop all five IQ sites and leave classic/gbm_hist.go:56. That single survivor must then be read in full before anything ships: confirm u is a bit pattern read from a byte slice rather than assembled arithmetically, and confirm the loop is hot enough to matter (gbm_hist is a histogram build, not a per-token path).

HONEST POSSIBILITY, to be stated rather than avoided: PS4001 may have ZERO true positives in this tree. If gbm_hist.go:56 also fails inspection, the correct outcome is not a cleverer discriminator but retiring or downgrading the check - a rule with 30 findings and no actionable site is pure noise, and its cost is paid by every future reader who triages it.

STILL OUTSTANDING from the earlier round, unchanged: inspect all suppressions individually rather than trusting the predicate definition (circular), and state little-endian as a REQUIREMENT ON THE FIX in the message - build tag or runtime byte-order check - since an unconditional bulk copy corrupts data on a big-endian host.

Nothing is applied. Tree is at the PS4001=30 baseline.

## T-01KYKSAF75FQGSFSQM9Z2RAJXQ crossentropy scalar math.Log: call the existing vlogF32 (gated on PROC-007)
kind: task
state: draft
created: 2026-07-28

FIRST GENUINE CANDIDATE among the recently triaged detector classes. Unlike PS4001 (30 findings, all false, fixed in d1ed762), PS3002 (preconditions unmet) and the integer half of PS5001, PS4002's cross-entropy sites are real: the replacement function already exists.

TARGET: backend/cpu/crossentropy.go:76 and :140 - math.Log running scalar in a loop. The file already calls vexpF32 and vexpRowF32, so the vectorized idiom is established locally, and vlogF32 / vlogQuadsNeonF32 are already in the configured vectorizedSiblingFuncs set. This is therefore CALL AN EXISTING HELPER, not write a new SIMD transcendental - a small change, which is what separates it from the PS4002 sites that would require authoring a vectorized primitive from scratch. Confirm vlogF32's signature and dtype before assuming it drops in; the crossentropy sites must be F32 for it to apply directly.

Also reported and NOT part of this task: backend/cpu/elementwise.go:273,:290 (math.Erf). No vectorized Erf exists in the configured set, so those sites mean "write one" - a materially larger job that should be judged separately.

GATED BY PROC-007/PROC-008. A vectorized log is a polynomial approximation, not math.Log, so this is NOT bit-identical - the same category the user just ruled on. Cross-entropy loss and its gradient are continuous outputs, so PROC-007 admits it in principle. But the standing gate applies: the two red nn ULP tests (TestGradFnBitIdenticalToSlowPath, TestEMAUpdateBitIdenticalToSlowPath) must be green first, so a deliberate approximation error does not land on top of an unexplained one and destroy attribution. Do not start until then.

WHEN UNBLOCKED, in this order:
1. Confirm entry: temporary panic at each math.Log site, run backend/cpu and nn tests, verify it fires. Do not read timings first - the AddBias lesson (benchmarks that name-match a feature but never reach it) and the gatherCast lesson both came from skipping this.
2. Establish the ERROR BOUND before the speed: compare vlogF32 against math.Log over the actual input domain of cross-entropy (probabilities in (0,1], where log is steepest near zero and worst-case relative error is largest). Report max ulp and max relative error. If the bound is not defensible near zero, reject - a loss that is wrong for confident-but-incorrect predictions is worse than a slow one.
3. Interleaved A/B, file-copy toggle, three alternations, medians, plus an unaffected control. Existing surfaces: BenchmarkCrossEntropyF32_256x4096_cpu and its backward twin; check they enter the changed branch per step 1.
4. Per PROC-008, replace any bit-identical assertion with a tolerance test whose bound is the one measured in step 2, and justify it in the commit. Prove non-vacuity by mutation, scoped to the test claiming the coverage.
5. Full-tree run to find every cross-path equality test the approximation disturbs - training-loop and reference-parity tests are the likely casualties.

NOT MEASURED: nothing was benchmarked, deliberately. Measuring before the numerics gate clears would invite shipping on a good number, which is the failure mode PROC-007 exists to prevent.

## R-01KYKSDHSPFH4STY7PBCNR83AA main is red: two nn bit-identity guards fail at a2f2746
kind: research
state: draft
created: 2026-07-28

MAIN IS RED. Two bit-identity guards fail on the mainline itself, not on any feature branch:
  nn/accum_fastpath_test.go:191   TestGradFnBitIdenticalToSlowPath    dt=f64 shape=(1) elem 0: fast -0.6544858867081182 vs slow -0.6544858867081184
  nn/weightavg_fastpath_test.go:267 TestEMAUpdateBitIdenticalToSlowPath dt=f32 shape=(7) param 0 elem 0: fast 0.06589753404259685 vs slow 0.06589753404259686

ATTRIBUTION, established rather than assumed: run directly in the main worktree at commit a2f2746, both fail with exactly these values. The docs/benchmark branch changes no Go source under nn - `git diff main -- nn/` shows only added .spectackle documents - so the branch is excluded as a cause, as is any uncommitted work in a sibling worktree. Both are 1-ULP divergences in the last decimal place, consistent with a summation-ORDER difference between a fast path and its scalar reference rather than a gross logic error.

SUSPECTS, by history of the implicated files: nn/accum_fastpath.go traces to 1ef3b8d (per-element hot-loop fast paths, batch 2, T896); nn/weightavg.go to ce866e9 (contiguous fast path for SWA/EMA Update, 4.2x / 3.1x) and also 1ef3b8d. Both are fast-path introductions, which is exactly where a reordered accumulation would enter. The accum test accumulates over several rounds before comparing, so an order difference compounds and only becomes visible after multiple Add calls - which is why a single-step check would miss it.

WHY THIS BLOCKS MORE THAN ITSELF: two separate optimization lines are gated on it by standing decision - T-01KYJYPQ38E93 (PS5001 reciprocal-multiply, including the hot attention-output normalization at backend/cpu/attn_extra.go:171) and T-01KYKSAF75FQG (crossentropy scalar math.Log to the existing vlogF32). Both are deliberate non-bit-identical changes under PROC-007. Landing either while these two guards are already red would make any future ULP divergence unattributable: no one could tell the deliberate approximation from the pre-existing defect.

TWO OUTCOMES, and the diagnosis must distinguish them: either a fast path genuinely reorders its accumulation and should be corrected to match the reference, or the guard is too strict for a reordering that was intended and should be a documented tolerance test under PROC-008. The second is only legitimate if the reordering was deliberate and is justified in writing - a test relaxed merely to go green would destroy the signal these guards exist to provide.

NOT TOUCHED: nn is the parallel worker's lane, so nothing here was edited or fixed - this record is diagnosis only, so the owner can act with the attribution already done.

## R-01KYKT7TPFF0NRNQ7162MZWX3H nn ULP failures root-caused: fast path does reciprocal-multiply, comment claims bit-identity
kind: research
state: draft
created: 2026-07-28

ROOT CAUSE IDENTIFIED. The failure is not a mystery defect: the fast path deliberately applies reciprocal-multiply, and its comment wrongly claims the result is bit-identical.

nn/accum.go:72-89, GradAccumulator.GradFn:
    k := float64(a.steps)
    ik := 1 / k   // "average by multiplying the invariant reciprocal, not dividing per element"
    ...
    d[i] = s[i] * ik
The slow reference (accum_fastpath_test.go:150-163) computes s[i] / k. These differ by up to a half ulp, because 1/k is rounded before the multiply. The comment at accum.go:81-83 asserts "the same sum/Steps average with the identical float32 rounding, so bit-identical" - that assertion is FALSE, and TestGradFnBitIdenticalToSlowPath is correctly reporting it. Observed at dt=f64 shape=(1): fast -0.6544858867081182 vs slow -0.6544858867081184.

This is the SAME transform as PS5001, which the ADR on non-bit-identical transforms was raised about and which PROC-007 now governs. The optimization was already present in the tree, undeclared, before that rule existed. TestEMAUpdateBitIdenticalToSlowPath in nn/weightavg_fastpath_test.go:267 fails with the same 1-ulp signature (f32, 0.06589753404259685 vs ...86) and should be checked for the identical cause - nn/weightavg.go traces to the same fast-path commits (1ef3b8d, ce866e9).

RESOLUTION IS DETERMINED BY THE RULES ALREADY ADOPTED, not by a fresh judgment call. This is ADR option (b), the deliberate-reordering branch:
1. The transform STAYS. Accumulated gradients and EMA weights are continuous outputs, which PROC-007 admits. Reverting a real optimization to satisfy a mistaken comment would be the wrong repair.
2. The COMMENT at nn/accum.go:81-83 must be corrected - it currently claims bit-identity the code does not deliver, which is how this went unnoticed. State the half-ulp reciprocal-multiply and cite PROC-007.
3. The GUARD becomes a tolerance test per PROC-008, with the bound measured over the actual accumulation depth (the error compounds across rounds, which is why a single-step check would pass and the multi-round test fails) and justified in the commit. It must NOT simply be relaxed until green - a bound derived from observation rather than analysis would hide the next real regression.
4. Same treatment for weightavg once its cause is confirmed identical.

WHY THIS MATTERS BEYOND THE TWO TESTS: these guards block the pre-push preflight hook, and two optimization lines are gated on them - T-01KYJYPQ38E93 (PS5001, including backend/cpu/attn_extra.go:171) and T-01KYKSAF75FQG (crossentropy vlogF32). Both are non-bit-identical changes of exactly this kind. Resolving these two guards correctly also establishes the pattern those tasks should follow.

NOT EDITED: nn is the parallel worker's lane and the standing instruction was to diagnose without editing. Everything needed to act is above - file, line, cause, and which of the two rules applies.

## T-01KYKTCRRZF7NRJTS862VAWBZV PS2004 per-call scratch: hoist to receiver fields where hot, sequential and fully overwritten
kind: task
state: draft
created: 2026-07-28

PS2004 (17 findings): a make() executed per iteration of a pointer-method loop, bound to a local that does not escape - per-call scratch reallocated on every call. The recommendation is to hoist it to a reused receiver field, growing on demand. This is an ALLOCATION class, so the metric is allocs/op and bytes/op, not wall time; a change here can be a clear win on garbage pressure while barely moving ns/op, and the benchmark must be read accordingly (b.ReportAllocs is already the harness default in this tree).

SITES OUTSIDE THE PARALLEL WORKER'S LANE, available to act on:
  classic/gmm.go:519
  classic/models.go:233
  linalg/linalg.go:106
  vision/swin.go:648
  vision/swin.go:773

SITES IN nn/nlp/autograd, NOT to be touched - parallel worker's lane:
  nlp/deepseekv2_decode.go:80
  nlp/deepseekv2_latent.go:136
  nlp/diffusion_lm.go:278
  nlp/t5_decoder.go:336
  nn/apollo.go:191
  nn/galore.go:214
  nn/memorizing_attention.go:379
  nn/muon.go:92
  nn/peer.go:289
  nn/qgalore.go:506
  nn/shampoo.go:96
  nn/spectral_norm.go:83

WHY THIS NEEDS MORE CARE THAN THE ODOMETER WORK: hoisting scratch into a receiver field changes object state, and that has two consequences the transform must answer before it ships.
1. CONCURRENCY. A receiver field turns a previously stateless method into one that mutates shared state. If any caller invokes the method concurrently on the same receiver - or the method is reached from a parallel worker pool, which several of these packages use - the hoist introduces a data race that no correctness test will reliably catch. Establish for EACH site whether the receiver can be shared across goroutines. Where it can, either skip the site or give each worker its own scratch; do not rely on "it looks sequential".
2. STALE DATA. The finding's own text warns to verify the buffer is fully overwritten before use. A hoisted buffer retains the previous call's contents, so any code path that reads before writing - a partial fill, an early break, a size that shrank since last call - silently consumes stale values instead of zeros. This is a correctness bug that reproduces only on the SECOND call, which is exactly the shape a single-call unit test misses. Prove full overwrite per site, or zero explicitly and measure whether the zeroing eats the win.

VERIFICATION SEQUENCE:
1. Confirm the loop is hot: these are fit/refit paths (gmm, spectral_norm, shampoo) and vision blocks, not per-token inference. If a site runs once per training run, the allocation saving is irrelevant and the site should be declined rather than optimized.
2. Prove entry with a temporary panic before reading any number - the standing lesson from this session's broadcastContig and gatherCast work.
3. Interleaved A/B with allocs/op and bytes/op as the primary metric, plus an unaffected control. Report ns/op too, but do not sell the change on it.
4. Correctness: a test that calls the method TWICE with different input sizes and asserts identical results to a fresh-receiver call. That is the test that catches stale-buffer reuse, and it must be shown to fail when the zeroing or overwrite is removed.
5. If the receiver can be shared, add a race-detector run over the touching tests.

DECLINE CRITERIA, stated so the answer can be "no": a site whose loop is cold, whose receiver may be shared, or whose buffer cannot be shown fully overwritten is not worth the state it introduces. Reducing allocations in a path that runs twice per process is not an optimization, it is added coupling.

## R-01KYM1WWMSF0GBMCJR1HVNPTFH PS2004 complete outside nn: 2 shipped, 3 declined by measurement
kind: research
state: draft
created: 2026-07-28

PS2004 triage COMPLETE outside the parallel worker's lane: 2 shipped, 3 declined, each decided by a measurement rather than by inspection.

SHIPPED
  linalg/linalg.go LU.Solve (bcf78a7) - forward-substitution scratch was allocated per right-hand-side column, so an n-column solve (what Inverse does) churned O(n squared) bytes to produce an O(n squared) result. 128x128: 135 -> 8 allocs/op, 393418 -> 263368 B/op. 64x64: 71 -> 8. Single-column control unchanged at 8, as it must be. Wall time within noise and NOT claimed.
  classic/gmm.go PredictProba (29fbe11) - per-sample log-responsibility scratch. 2048x8: 4098 -> 2051 allocs/op, 311360 -> 180352 B/op. The win depends on K, not on sample count: present at k=8 for both 512 and 2048 samples, absent at k=4 for both. Mechanism not established, claim limited accordingly.

DECLINED, with the number that decided it
  classic/models.go:344 (Armijo line-search scratch) - saves 3 allocations out of 1263. The benchmark shows it: s50 and s200 report identical allocation counts, so the Newton loop converges in far fewer than 50 steps and the per-step scratch is allocated about three times per fit. Benchmark kept and committed (dde27fa) since Fit had no measurement surface; the optimization reverted.
  vision/swin.go:773 - make([]*tensor.Tensor, 4), 32 bytes, inside a loop whose body does a tensor slice, four gathers and a concat. The allocation is noise against the kernel work in the same iteration.
  vision/swin.go:648 - same shape, same reasoning.

TWO DESIGN DECISIONS worth carrying to the remaining nn sites:
1. FUNCTION LOCAL, NOT RECEIVER FIELD. PS2004's text recommends hoisting to a receiver field. Both shipped sites instead hoisted to a local outside the loop, which captures the identical allocation saving without turning a read-only method into one that mutates shared state. LU factorizations are immutable after Factor and GaussianMixture models are read-only during PredictProba, so both are safe to call concurrently today; a receiver field would have broken that silently. Prefer the local wherever the scratch's lifetime is one call.
2. NOT EVERY SIBLING ALLOCATION CAN MOVE. In PredictProba the neighboring o escapes into out[i] and each sample genuinely needs its own; hoisting it would be a correctness bug. Check escape per allocation, not per loop.

THE TEST THAT MATTERS is the multi-item-versus-single-item comparison, not a correctness spot check. TestLUSolveMultiColumnMatchesPerColumn pins a multi-column solve bit-identically against solving each column separately with a FRESH factorization. Stale-buffer reuse only shows from the second item onward, so a single-item test is structurally blind to it. Non-vacuity confirmed by mutation (leaving y[0] from the previous column turns it red).

DETECTOR LEFT UNCHANGED, deliberately. PS2004 was correct at every site including the three declined ones - it found real per-iteration allocations, and whether they matter is a question of iteration count and surrounding work that an AST cannot answer. Contrast PS4001 (fixed in d1ed762), where the recommended transform was invalid at every site and a suppression was warranted. A rule whose findings need measurement is not a defective rule.

REMAINING: all other PS2004 sites are in nn/nlp/autograd (apollo, galore, memorizing_attention, muon, peer, qgalore, shampoo, spectral_norm, deepseekv2 x2, diffusion_lm, t5_decoder). Untouched by design - parallel worker's lane. The two design decisions above apply directly to them.

## R-01KYM1ZJRHFZVBYGF7G1A33DMJ PS2002 declined outside nlp: every reachable site is debug-gated or one-shot
kind: research
state: draft
created: 2026-07-28

PS2002 (13 findings) DECLINED WHOLESALE outside the parallel worker's lane. Not a detector defect - the recommendation is sound and its fix (.Grow(n) before the write loop) is semantically inert, so this class would normally be the safest thing to act on. It fails on leverage alone: every reachable site is cold by construction.

SITES OUTSIDE nn/nlp/autograd, all four declined:
  tensor/shape.go:74, Shape.String() - the only two call sites in the tree are backend/execute.go:98 and :122, and BOTH sit behind debug flags (debugFallback, debugTimeOps). They execute only when a debug env var is set, so the builder never runs in a normal process. Verified by reading the call sites, not inferred from the name.
  internal/npy/npy.go:265, :303, :319 - all inside Write, one-shot file serialization. The header string is built once per file written.

REMAINING NINE are in nlp (grammar.go, jlens_viz.go x2, jsonschema.go x5, tiktoken.go) - untouched by design, parallel worker's lane. Worth noting for whoever owns them that jsonschema.go carries five of the nine, so that file alone is where the class concentrates; and tiktoken.go:69 is the one site of the nine plausibly on a per-encode path rather than a one-shot setup path, so it is the only one likely to repay measurement.

METHOD POINT, cheap and repeatedly decisive this session: for a class whose fix is safe, hotness is the ONLY question, and it is usually answerable by reading call sites rather than by benchmarking. Two greps settled all four sites here. Benchmarking a debug-gated or one-shot path only produces noise, and a passing benchmark on a cold path is the most misleading artifact available - it looks like evidence.

DETECTOR UNCHANGED. PS2002 correctly identifies unsized builders; whether a given one is worth pre-sizing depends on call frequency, which an AST cannot see. Same disposition as PS2004 (R-01KYM1WWMSF0G) and the opposite of PS4001 (d1ed762), where the transform itself was invalid. Do not add a suppression here - the findings are true, they are simply not urgent.

## R-01KYM3MB5MERB82PHQ248XN0G0 PS4006 complete outside nn: 4 sites shipped, and two kernels had no correctness tests
kind: research
state: draft
created: 2026-07-28

PS4006 (row-slice matrix layout) COMPLETE outside the parallel worker's lane: four sites, all measured, all shipped, all now self-silencing.

  backend/ref/solvespd.go   2.15x  allocs 146 -> 18   (25c47a0)
  backend/ref/cholesky.go   1.50x  allocs 137 -> 10   (1ba1b51)
  backend/ref/qr.go         1.35x  allocs  98 -> 34   (bcf9e13)
  internal/linalg SymEig    1.20x  allocs 277 -> 149  (e5a8053)

All four are bit-identical by construction: index arithmetic only, same operands and same order. The defect is layout, never arithmetic — a [][]T matrix pays one heap allocation per row, and any column walk (m[k][p] with k varying) then dereferences a different row pointer per step.

DETECTOR PROVENANCE, worth noting for how the loop is supposed to work: the first two sites were found by hand, and no existing rule could see them - PS1001 pointed at the O(n²) accessor reads in cholesky while the O(n³) row-pointer walk beside it went unmentioned, and PS2004 sees per-iteration allocations but not the layout that makes them costly. Only after TWO independent measured instances was PS4006 written (904a467), deliberately not after one, to avoid fitting a detector to a single site. It then found QR and SolveSPD itself, and both paid. Self-silencing was verified for every site: applying the rule's advice removes the finding.

THE LARGER FINDING IS THE TEST GAP, not the speed. QR and SolveSPD had NO correctness coverage at the level their rewrites touched. Deliberate mutations - a transposed index in the Householder update, a one-ulp perturbation of the forward substitution - passed the entire backend/ref suite, and for SolveSPD the autograd suite too. Both rewrites would have shipped unguarded and undetectably wrong.

Property checks are the trap that allowed it. Q.R == A and QtQ == I hold to tolerance under exactly the drift being guarded against, so a suite built on them cannot distinguish a correct rewrite from a subtly broken one. Each kernel now keeps its ORIGINAL implementation as an in-test oracle and compares bit for bit; for SolveSPD the oracle takes L from the same Cholesky the kernel uses, so the two cannot diverge on the factorization rather than the substitution.

Cast as PROC-009: probe the suite with a one-ulp mutation before rewriting a kernel, and if it survives, write the oracle BEFORE optimizing. Order matters - writing it first is what proves the oracle encodes the old behavior rather than the new.

REMAINING PS4006 SITES are in nn/nlp/autograd (vjp_cholesky, vjp_eigh, vjp_qr, vjp_logdet, gradcheckapi and others - 71 findings tree-wide). Untouched by design. Whoever takes them should expect the same test gap: the vjp_* files mirror the kernels whose coverage was just shown to be absent, so PROC-009 applies directly.

backend/ref/eigh.go still reports because SymEig's exported signature takes [][]float64. Flattening that is an API change across internal/linalg, not a local edit, and was deliberately not bundled here.

## R-01KYM40K9NFRNVTYDYE5D3MXP9 PS3003 resolved outside nn: einsum 5x shipped, four sites declined on key density
kind: research
state: draft
created: 2026-07-28

PS3003 (integer-keyed map read in a loop) resolved outside the parallel worker's lane: ONE large win, the rest declined on key density or hotness.

SHIPPED - backend/einsum.go, 4.5-5x (d58e40b). Both maps were map[byte]int, and byte keys make a [256]int a DENSE and EXACT replacement: the bound is the key type itself, so the rule's density precondition is satisfied by construction rather than by inspection. `val[ix] = rem % size[ix]` inside the O(total) contraction loop was two map operations per index per combination, which is why hashing dominated the engine. A parallel [256]bool preserves the comma-ok distinction between absent and zero. Control benchmark added afterwards and the claim retro-verified at 4.60x against a control flat at 1.00x.

DECLINED, with the reason that decided each:
  classic/tree.go:833, :951 - map[int]int over class LABELS, which are arbitrary integers, not dense over [0,N). A slice replacement would need a min/max range check and a fallback, and the loop runs once per Fit over n samples, not per prediction. The rule's own precondition is unmet.
  classic/forest.go:198 - pos[label] over tree.classes, a handful of entries per tree. Cold.
  format/pytorch/pytorch.go:172 - inside unpickler.run, one-shot model loading.
  internal/perfscan/perfscan.go:933, :937 - the scanner's own tables, run once per file.

THE GENERALIZABLE POINT, worth applying to the remaining nn/nlp sites: PS3003's density precondition is answerable from the KEY TYPE alone in the best case. A map keyed by byte or by a small enumerated type can always become an array, no inspection needed; a map keyed by an unconstrained int cannot without a range analysis the AST does not have. Sorting the remaining findings by key type first would separate the mechanical wins from the ones needing judgment, and would cost one grep.

DETECTOR UNCHANGED. Every finding was a real integer-keyed map read in a loop; whether it is worth replacing depends on key density and call frequency, neither of which an AST can settle. Same disposition as PS2004 and PS2002, and unlike PS4001 where the recommended transform was invalid at every site.

TEST GAP, the third of the session: a one-ulp perturbation of the einsum contraction product passed the entire backend suite. EinsumContract had no correctness coverage at the level the rewrite touched, exactly as QR and SolveSPD did not. PROC-009 now requires probing for this before rewriting, and it was followed here — the oracle was written and passing against the original implementation before any change.

## R-01KYM4EDWPEFYVQAJZ16PJJWJ8 Five of five numerical kernels are blind to a one-ulp change - tolerance testing is the cause
kind: research
state: draft
created: 2026-07-28

FIVE of five numerical kernels probed this session were blind to a one-ulp change. This is not a gap in coverage; it is a structural property of how they are verified, and it matters more than any individual speedup found alongside it.

THE FIVE, each probed by inserting a deliberate one-ulp perturbation and running the full owning suite plus autograd:
  backend/ref/qr.go            no test at this level; two index mutations also passed
  backend/ref/solvespd.go      no test at this level
  backend/einsum.go            no test at this level
  backend/ref/svd.go           no test at this level
  linalg/derived.go Pinv       HAS a dedicated test file, derived_test.go, and still passed

The last is the important one. Pinv is not untested — it is tested to a TOLERANCE. So the finding is not "these kernels lack tests" but "this repo verifies numerical kernels by tolerance, and tolerance is structurally blind to the defect class that layout and dispatch rewrites introduce". A rewrite that transposes an index, reorders an accumulation, or reads a stale buffer produces a result that is still close to correct, and every property or tolerance assertion continues to pass.

WHY PROPERTY CHECKS ARE THE TRAP, stated concretely so it is not mistaken for a style preference: Q·R == A and Qᵀ·Q == I hold to tolerance under exactly the drift being guarded against. A suite built on them cannot distinguish a correct rewrite from a subtly broken one — demonstrated, not asserted: QR's suite passed both a transposed Householder index and a one-ulp perturbation of the Q accumulation.

WHAT WORKS, used for all five: keep the ORIGINAL implementation as an in-test oracle and compare raw bits (math.Float64bits, no tolerance). It is more code than a property check and it is the only thing that catches this class. Two further details earned by experience — write the oracle BEFORE the optimization, which is what proves it encodes the old behavior rather than the new (PROC-009); and where the kernel depends on another (SolveSPD on Cholesky, Pinv on SVD), take the dependency's output from the SAME call the kernel uses, so a divergence cannot be blamed on the dependency.

SCOPE NOT YET COVERED: only kernels reached while optimizing were probed. backend/ref alone holds roughly forty more, and the vjp_* files in autograd mirror precisely the kernels just shown to be unguarded. A systematic ULP-probe sweep of backend/ref would cost one mutation and one test run per kernel and would produce a ranked list of what is unverified — plausibly the highest-value use of a session that is not itself an optimization.

NOT A PERFSCAN RULE, deliberately. Test sensitivity is not an AST property: no static pattern distinguishes a bit-exact assertion from a tolerance one in a way that would survive contact with real test code. The mutation probe is the instrument, and PROC-009 is where it belongs.

## R-01KYM4HGM1EEY8RBVWGAWX2PV6 ULP audit of backend/ref: 11 of 12 kernels blind to a one-ulp change, flashattn among them
kind: research
state: draft
created: 2026-07-28

SYSTEMATIC ULP-BLINDNESS AUDIT of backend/ref, run with a scripted mutation probe: insert `* 1.0000000000000002` on the first float accumulation in a kernel, run the owning suites, record whether anything goes red, restore. Twelve kernels probed. ELEVEN are blind. One is guarded.

BLIND (a one-ulp change in the accumulation passes every test):
  backend/ref/flashattn.go     s += qv * krow[d]                          <- attention core
  backend/ref/crossentropy.go  v += zl * lse * lse
  backend/ref/conv.go          acc += xrow[ix] * wrow[kx]
  backend/ref/distill.go       kl += p[j] * (math.Log(p[j]) - math.Log(q[j]))
  backend/ref/grpo.go          total += surr - beta*kl
  backend/ref/cpo.go           total += softplus(-z) + alpha*(-lw)
  backend/ref/ipo.go           total += d * d
  backend/ref/qr.go, solvespd.go, svd.go, and backend/einsum.go - found earlier while optimizing
  linalg/derived.go Pinv - has a dedicated test file and is still blind

GUARDED: backend/ref/gemm.go only. Worth reading to see what it does differently; it is the single existing example of the pattern the other eleven need.

SKIPPED, not cleared: backend/ref/logdet.go and conv_backward.go matched no float accumulation under the probe's heuristic (first line of the form `x += a * b`). They are unprobed, not verified.

WHAT THIS MEANS: flashattn is the one that should worry someone. Attention is the hottest path in the library and its reference implementation cannot detect a one-ulp error in the QK inner product — which is exactly the error a devirtualization, a reordering or a stale-buffer bug produces. The five optimizations shipped this session into unguarded kernels were each given a bit-exact oracle first (PROC-009); the other seven kernels above have no such protection and no optimization has touched them yet.

METHOD, reusable: the probe script is ~20 lines - regex the first `x += a * b`, append the ulp factor, run the suite, diff on FAIL, restore. It found eleven unverified kernels in two batched runs. Any future session can rerun it over a wider file list; the cost is one test run per kernel.

PRIORITY ORDER for adding oracles, by how hot the kernel is and how likely a rewrite is to touch it: flashattn, conv, crossentropy, then the preference losses (grpo, cpo, ipo, distill) which share the DPO/KTO/PPO shape already optimized this session.

NOT A PERFSCAN RULE: test sensitivity is not an AST property. The mutation probe is the instrument and PROC-009 is the standing requirement; this audit is the inventory it should be applied against.

## R-01KYM5954REZKTA5XR5TV0R2FD PS4006 declined for naivebayes: row-pointer hoisting is the cheap mitigation
kind: research
state: draft
created: 2026-07-28

PS4006 DECLINED for classic/naivebayes.go (4 sites), and the reason generalizes into a cheap mitigation worth knowing before reaching for a flatten.

WHY DECLINED: the hot path already hoists the row pointer. GaussianNB.jointRow does
    ln := m.logNorm[c]
    iv := m.invSigma[c]
and then indexes ln[j], iv[j] in the inner loop. The row dereference is paid ONCE per class per query row rather than once per element, which is the entire cost the flatten removes. Flattening would buy the allocation count and nothing else, on a path whose allocations are one-time.

WHAT PS4006 ACTUALLY FLAGGED at :144, :157, :181, :182 is the CONSTRUCTION loop — `logNorm[c][j] = ...` inside `for c { for j {` — which is the fit path, executed once per Fit, not per prediction. The two-deep index there is unavoidable: filling a [][]T requires it.

DETECTOR OBSERVATION, not yet a fix: every [][]T necessarily has an allocation loop followed by a fill loop with a two-deep index, so PS4006 will fire on that pair even when every READ hoists the row correctly. The four sites shipped this session (cholesky, SymEig, QR, SolveSPD, SVD) were genuine because their HOT loops indexed two-deep, not merely their construction. A refinement would require the two-deep index to appear in a loop nest other than the one that fills the matrix — but note this would NOT have suppressed any of the five true positives, since in those the fill and the hot loop are the same nest shape. Worth measuring the finding-count delta before implementing; the risk is losing a true positive whose fill loop IS the hot loop.

THE MITIGATION, generally useful: hoisting `row := m[i]` above the inner loop captures most of the flatten's benefit without changing the type or touching any caller. It leaves the allocation count alone, so it is the right move when allocations are one-time and the reads are hot — exactly naivebayes's shape. Reach for the flatten when the allocation count also matters (SolveSPD: 146 -> 18) or when the access is a COLUMN walk, which hoisting cannot help because each step needs a different row.

NOT MEASURED: no benchmark was run. Declining on structure was sufficient here — the row dereference is provably outside the inner loop by reading five lines, and a measurement would only confirm what the code already shows.

## T-01KYM5BJANE63RPY53Z7QDFSRP Bit-exact oracle for ref/flashattn — the Metal parity test depends on it
kind: task
state: draft
created: 2026-07-28

Write a bit-exact oracle for backend/ref/flashattn.go. It is the top item of the ULP audit (R-01KYM4HGM1EEY) and the risk compounds in a way the audit did not yet capture.

WHY IT IS WORSE THAN A FALLBACK. backend/cpu overrides OpFlashAttn, so ref's version is not the production path — but backend/metal/metal_test.go:335 validates the Metal FlashAttention-2 kernel AGAINST IT. ref/flashattn is the correctness reference for the GPU implementation. It is blind to a one-ulp change in its QK inner product (`s += qv * krow[d]`), so the GPU parity test inherits that blindness: a reference drift and a GPU drift of the same magnitude would cancel in the comparison and neither would be reported.

WHY THIS ORACLE IS HARDER THAN THE SEVEN ALREADY WRITTEN, and must not be done the easy way: flash attention's defining property is that it computes the SAME mathematical result as naive attention by a DIFFERENT accumulation order — blocked online softmax with running max and rescaling. A naive attention oracle is therefore NOT bit-identical to it and would have to be compared with a tolerance, which is exactly the failure mode this whole line of work exists to close. The oracle must reproduce the kernel's own block sequence: same block size, same running-max update, same rescale points, same causal masking, in the same order.

APPROACH: copy the kernel's loop structure into the test as the oracle, as was done for QR (bcf9e13), SymEig (e5a8053), SolveSPD (25c47a0) and SVD (6e961d9). It is more code than the others because the nest is deeper — heads, blocks, rows, then the inner dk loops at flashattn.go lines 77-140 — but the technique is identical and the resulting test is the only kind that can catch a reordering.

VERIFY: cover causal and non-causal, at least two block sizes including one that does not divide seq evenly, seq shorter than one block, dk of 1, and multiple heads. Then mutate, choosing probes that match the defect class per PROC-010: transpose an index in the QK product, perturb the running max, drop a rescale. A one-ulp scale on a term the running max later dominates may legitimately be absorbed — explain such a green rather than treating it as a weak test.

SCOPE NOTE: this is a correctness task, not an optimization. No benchmark is required and no behavior should change. If a measurable win appears while reading the kernel, it belongs in a separate task with its own A/B — do not bundle a rewrite with the test that would guard it, since the test must be shown passing against the CURRENT implementation first (PROC-009).

## R-01KYM5E3PCEETVZA541J0N8WV1 PS1002/PS1003/PS2001/PS3001 closed outside nn: real findings, inverted cost/benefit
kind: research
state: draft
created: 2026-07-28

The four small perfscan classes (PS1002 alloc-in-loop, PS1003, PS2001, PS3001 batch-single-elt) are closed outside the parallel worker's lane. Seventeen findings tree-wide, three of them outside it, none actionable.

rl/ppo.go:248 and rl/rl.go:203 (PS3001) — `forward(backend.NewContext(), net, [][]float64{obs})` inside the env-step loop. The finding is real: a [][]float64 wrapper is allocated per step to carry one observation into a batch API. It is declined on leverage. Each iteration runs a full nn.Sequential forward pass — matmuls over the whole network — plus a separate OpSoftmax execute. A slice header against that is noise.

The rule's recommended fix also costs more than it returns here: "call a single-item API instead" presumes such an API exists, and nn.Sequential exposes only the batch form. Acting on this site means ADDING a single-item forward path to nn — a new public surface, in the parallel worker's package, to save one small allocation per env step. The cost/benefit is inverted, and the site should stay reported rather than suppressed: on a different caller with a cheap body the same finding would be worth acting on.

llamagpu/decoder.go:3148 (PS1002) — a Cast allocation inside a per-element loop. Not evaluated: llamagpu is CUDA-facing and the parallel worker owns that area. Left untouched by lane discipline, not by judgment on the finding.

DETECTORS UNCHANGED for all four. Every finding was structurally correct; what disqualified the rl sites is the ratio between the flagged allocation and the surrounding work, which an AST cannot weigh. Same disposition as PS2002 and PS2004 — and worth stating plainly, since three classes in a row have now been closed on leverage rather than on correctness. That is the expected shape for a mature detector set: the rules find real patterns, and measurement decides which ones pay.

METHOD NOTE: no benchmark was run for the rl sites. Writing one would require an env harness, and the structural argument is decisive without it — the allocation is a slice header, the loop body is a neural network forward pass. Declining on structure is legitimate when the ratio is that lopsided; it would not be if the two were within an order of magnitude.

## R-01KYM5J5Z8EK99B9JWWNR0E2X9 Collapse tests guard mha; the same trick does NOT work for flashattn (measured)
kind: research
state: draft
created: 2026-07-28

Two findings for the flashattn oracle task (T-01KYM5BJANE63), one a technique worth reusing and one a refuted shortcut that would otherwise be tried first.

THE TECHNIQUE — why backend/ref/mha.go is GUARDED while flashattn is blind. Probing mha's QK product turns TestMHAMaskedCollapsesToMHA and TestMHASelectCollapsesToMasked red. Those are COLLAPSE tests: they run a more general operator in a degenerate configuration where it must equal a simpler sibling, and compare the two implementations against each other. No oracle is duplicated in the test — an existing sibling IS the oracle. That is far cheaper to write and to maintain than copying a kernel's loop nest, and it is the reason mha is the second of only three guarded kernels found in eighteen probes (with gemm and, by construction, the seven given oracles this session).

THE REFUTED SHORTCUT, measured rather than assumed. The obvious application to flashattn is OpFlashAttn with Block >= seq against OpMHA: with a single block the online softmax should reduce to an ordinary one. It does not agree bitwise. Measured over seq=6, heads=2, dk=4: 27 of 48 elements differ for causal=false, 21 of 48 for causal=true. The cause is structural, not a block-count artifact — flash accumulates an UNNORMALIZED weighted sum and divides by the running denominator at the end, while MHA normalizes the softmax first and then multiplies. Different operation order, different rounding, regardless of block count.

CONSEQUENCE for T-01KYM5BJANE63: the collapse approach is unavailable and the task's original specification stands — reproduce the kernel's own block sequence in the test, as was done for QR, SymEig, SolveSPD and SVD. Anyone reaching for the cheaper route should read this first; the experiment costs a few minutes and the result is unambiguous.

GENERAL RULE OF THUMB from the pair: prefer a collapse test when a sibling implementation exists AND the two agree bit-exactly in the degenerate case — verify that agreement before building on it, since algebraic equivalence does not imply bitwise equivalence. Fall back to duplicating the loop nest only when it does not.

AUDIT RUNNING TOTAL, eighteen kernels probed: GUARDED gemm, mha. BLIND flashattn, crossentropy, conv, distill, grpo, cpo, ipo, retention, zloss, blas1, qr, solvespd, svd, einsum, Pinv. SKIPPED (unprobed, not cleared) logdet, conv_backward, ia3, embed. blas1 is worth noting alongside flashattn: it holds the dot-product primitives, so it is foundational rather than peripheral.

## R-01KYM6C5AWFKZ8VP8QHGB3XTVA Production cpu/mha is ULP-blind while its reference twin is guarded
kind: research
state: draft
created: 2026-07-28

The ULP blindness extends to the PRODUCTION backend, and in the worst place: backend/cpu/mha.go is blind while its reference twin backend/ref/mha.go is GUARDED. The implementation that actually runs is less verified than the one it is checked against.

MEASURED, three independent mutation points in cpu/mha.go, each a one-ulp scale, each run against the full backend/cpu suite:
  line 351  s += float64(qv) * float64(kr[d])          QK inner product        BLIND
  line 596  dot += float64(pr[j]) * float64(dar[j])    backward accumulation   BLIND
  line 338  s += g.slopes[h] * float64(j-(g.off+i))    ALiBi position bias     BLIND
By contrast, the same probe on backend/ref/mha.go turns TestMHAMaskedCollapsesToMHA and TestMHASelectCollapsesToMasked red (R-01KYM5J5Z8EK9). The collapse tests guard the REFERENCE; nothing equivalent guards the production path.

WHY THIS INVERTS THE USUAL ASSUMPTION: cpu overrides both OpMHA and OpFlashAttn, so ref/mha is a fallback that most callers never reach, and cpu/mha is what every attention call on this host executes. Verification effort has landed on the wrong side of that split. The backward accumulation at 596 is the sharper half — a gradient defect degrades training silently rather than producing an obviously wrong forward output.

HOW IT WAS FOUND: PS6001 (012326b, extended in 727ef7b) began reporting backend/cpu once the dtype-switch form was covered, which is what prompted probing the production backend at all. The earlier audit (R-01KYM4HGM1EEY) had sampled only backend/ref, so its seven-for-seven result understated the exposure — it measured the reference implementations, not the ones that run.

WHAT IS NOT YET KNOWN: only mha was probed in cpu. elementwise, conv and crossentropy returned SKIP because the probe's heuristic (first line matching `x += a * b`) found no match in them; they are UNPROBED, not cleared. A targeted probe per file would settle them, and PS6001 lists 56 dual-path functions tree-wide as the population to work through.

RECOMMENDED ORDER: cpu/mha forward and backward first, since attention is the hottest path and the backward is the least visible; then the remaining cpu dual-path functions; then ref. The collapse-test technique that guards ref/mha may transfer directly — cpu/mha against ref/mha in a shared configuration — but per R-01KYM5J5Z8EK9 that agreement must be MEASURED before being relied upon, since the flashattn collapse looked equally plausible and failed in 27 of 48 elements.

## R-01KYM78A9BEBCRJHERXPEBAP68 Correction: cpu/mha:596 is unreached, not blind — audit evidence overstated
kind: research
state: draft
created: 2026-07-28

CORRECTION to R-01KYM6C5AWFKZ, and a revised count for the production attention audit.

WHAT WAS WRONG: that record listed backend/cpu/mha.go:596 (mhaBwdGemmBand) as BLIND on the strength of a surviving one-ulp mutation. The measurement was vacuous. A panic probe shows mhaBwdGemmBand is not reached by the backend/cpu suite at all, nor by an f32 backward at seq=128. It is selected by `case f32NativeKernels && seq >= mhaGemmMinSeq` — the SIMD experiment build described in CPU-002 — so in the default build it is dead code, and mutating dead code always survives. Unreached is not unguarded.

REVISED STATUS of the three probe points originally reported:
  mha.go:351 QK product          genuinely blind; now guarded bit-exactly (3d3e882, 92320a2)
  mha.go:338 ALiBi bias (f32)    genuinely reached; guarded against a sign flip. A one-ulp scale is
                                 absorbed because the bias is small beside the score (PROC-010), and a
                                 +1 distance shift stays green and is still UNEXPLAINED
  mha.go:596 GEMM-band backward  NOT blind — unreached in the default build. Requires the experiment
                                 build to test at all
The audit's headline claim — that production attention was less verified than its reference — stands on 351 alone, which is sufficient, but the evidence was overstated by one third.

WHAT NOW GUARDS PRODUCTION ATTENTION:
  forward F64 and F32   bit-exact against ref (0 of 48 measured before relying on it)
  backward F64          bit-exact against ref; catches mutations at mha.go:660 and :675
  backward f32          tolerance rel 2e-3 / abs 1e-4, margins 1.53e-04 and 2.38e-07. The bound is
                        BORROWED from CPU-002, which scopes it to the experiment-build GEMM path, not
                        mandated for this one
  mhaBwdGemmF32         untested and untestable without the experiment build

CAST AS PROC-012: confirm the mutated line executes before concluding a probe means the code is unguarded. This is the missing half of PROC-009 — that rule says to probe before rewriting, but says nothing about validating the probe itself, which is exactly where both errors happened. The other was the AddBias benchmarks, which measured a broadcast path that bcastBlockApply shadows, and that one manufactured an entire investigation before the control benchmark exposed it.

METHOD NOTE: the correction surfaced only because a sign-flip probe on 596 came back green AFTER a tolerance test was in place, which was inconsistent with the earlier BLIND reading. Two probes disagreeing is the signal worth chasing; a single probe agreeing with expectation is not evidence.

## R-01KYM7SSDRETETAM7BHRGR5CXM Leverage heuristic corrected: cpu registration can be dtype-partial and build-gated
kind: research
state: draft
created: 2026-07-28

CORRECTION to the leverage heuristic used throughout this session, and to two triage decisions that rested on it.

THE HEURISTIC WAS: if backend/cpu registers an op, ref's version is a fallback most callers never reach, so optimizing ref is low leverage. That is TRUE for most ops but WRONG in two ways, both discovered by accident while fixing a vacuous cross-entropy parity test.

(1) REGISTRATION CAN BE DTYPE-PARTIAL. cpu/elementwise registers OpSoftplus and OpSoftCap for F64 ONLY. In F32 those ops fall through to ref, so ref is production for them.

(2) REGISTRATION CAN BE BUILD-GATED. Inside `if vexpF32Fast` — the SIMD perf build — cpu registers OpGELUBackward and OpSiLUBackward for F32, and cpu/crossentropy registers OpCrossEntropy and OpCrossEntropyBackward for F32. In the DEFAULT build none of those exist and every one falls back to ref.

WHY IT MATTERS: the registration comment states gelu_backward was 18.9 percent of the f32 GPT training step, and silu_backward is the SwiGLU-FFN VJP on every Llama/Qwen/Mistral layer. In the default build both run ref's scalar implementations. These are not peripheral paths.

WHAT THIS DOES NOT IMPLY: there is no work to do here. The fast kernels already EXIST in cpu and are deliberately gated, because vectorizing a transcendental is not bit-identical and rides the ADR-0021 f32 tolerance. The leverage is real; the optimization is written; the gate is a policy decision about numerics, not an oversight. Acting on it means enabling a build, not writing code — and that is the operator's call, not an agent's.

DECISIONS THIS CORRECTS: the PS4003 triage concluded the ref/elementwise transcendental sites were "gated, therefore not actionable", and the PS4002 triage treated ref as fallback for the same family. Both conclusions survive, but for a DIFFERENT reason than stated — not because ref is unreachable, but because the replacement is a numerics trade already made deliberately elsewhere. The original wording would mislead a reader into thinking those paths are cold. They are not.

METHOD, going forward: to decide whether ref is production for an op, read cpu's init() for that specific op and check BOTH the dtype and any enclosing conditional. `grep -l "OpFoo" backend/cpu` is not sufficient and was the basis of several judgments in this session. Only crossentropy.go and elementwise.go carry such gates today, so the other leverage calls stand — verified rather than assumed.

## R-01KYM8N2N2E569XCGA0V5P6J2R Audit correction: ULP probe used too narrow a test scope — most BLIND verdicts were wrong
kind: research
state: draft
created: 2026-07-28

MAJOR CORRECTION to R-01KYM4HGM1EEY. Its headline — 11 of 12 backend/ref kernels blind to a one-ulp change — is WRONG, because the probe ran the wrong test scope.

THE ERROR: the probe script ran `go test ./backend/ref/ ./backend/` after each mutation. The cross-reference tests that guard ref kernels live in ./backend/cpu/ and ./backend/metal/, which were never run. Any ref kernel guarded from a sibling package was therefore recorded as blind.

RE-AUDITED at scope ./backend/... ./linalg/ ./nn/:
  backend/ref/flashattn.go      GUARDED by TestFlashAttnRetentionMatchRefWithinUlps (backend/cpu)
  backend/ref/crossentropy.go   GUARDED
  backend/ref/conv.go           GUARDED by TestConvCrossReferenceExact (backend/cpu)
  backend/ref/cumsum.go         GUARDED
All four were reported BLIND. flashattn was the audit's headline priority and the subject of T-01KYM5BJANE63, which is now UNNECESSARY and should be closed: the existing test catches a one-ulp change in the QK product, which is exactly what the proposed oracle would have caught, and R-01KYM5J5Z8EK9 already recorded that the cheap collapse route does not work. That task would have spent significant effort duplicating coverage.

WHAT SURVIVES, and why the distinction matters:
  - The seven kernels given oracles this session (qr, solvespd, svd, einsum, Pinv, distill, blas1, zloss, retention) were each probed at a scope that DID include their owning package, and several were confirmed unguarded by mutations that passed everything. Those findings stand, though flashattn's case means each deserves a re-probe at full scope before being cited again.
  - R-01KYM6C5AWFKZ (production cpu/mha blind while ref/mha guarded) STANDS. Those probes ran ./backend/cpu/, the correct owning scope, and the MHA guards were separately confirmed to add coverage: with mha.go:351 mutated, the pre-existing TestCPUCrossReferenceExact, TestConvCrossReferenceExact and TestMHA* all stay green while the new guard goes red.

ROOT CAUSE, and it is not the scope alone: the probe script was written to be fast, and a narrow scope makes each iteration cheaper. That trade silently converted "not covered by these packages" into "not covered", which is the same unreached-versus-unguarded confusion PROC-012 addresses at the line level. The scope version belongs beside it: a mutation probe proves nothing about tests it did not run.

CONSEQUENCE FOR THE REPO: this codebase is better verified than the audit claimed. The cross-reference convention (TestXxxCrossReferenceExact, TestXxxMatchRefWithinUlps) is used more widely than a ref-package-local search reveals, and PROC-014 now requires searching for it before writing a parity guard.

## R-01KYM9GSH7F5CR0MGGRYNWWDMN Close T-01KYM5BJANE63 no-action: flashattn is guarded; audit premise was false
kind: research
state: draft
created: 2026-07-28

CLOSES T-01KYM5BJANE63 with no action. That task specified a bit-exact oracle for backend/ref/flashattn.go, on the audit's claim that it was blind and that the Metal parity test inherited the blindness. Both premises are false.

flashattn is GUARDED by TestFlashAttnRetentionMatchRefWithinUlps in backend/cpu, which turns red on a one-ulp perturbation of the QK inner product — precisely the defect the proposed oracle was to catch. The audit missed it because its probe ran only ./backend/ref/ and ./backend/, never the cpu package (R-01KYM8N2N2E56, now PROC-015).

The task would have been expensive to satisfy and was already known to resist the cheap route: R-01KYM5J5Z8EK9 measured that flash-with-one-block does not collapse onto MHA bitwise (27 of 48 elements differ, causal false; 21 of 48, causal true), so the oracle would have had to reproduce the blocked online-softmax sequence by hand across heads, blocks, rows and dk. That is the largest single piece of work this session's correctness line proposed, and it was unnecessary.

WHAT THE MISTAKE COST AND WHAT IT DID NOT: it cost two iterations of correction and ten tests written then removed. It did not cost any wrong code — every optimization shipped was validated by interleaved A/B with an unaffected control, independent of the audit. The failure was confined to the verification inventory, which is where an overstatement is cheapest to hold and cheapest to fix.

WHAT SURVIVES from the audit line, all measured at correct scope: production backend/cpu/mha.go was genuinely unguarded for the QK product and the backward accumulation, and is now guarded bit-exactly in both dtypes forward and in F64 backward, with the f32 backward covered by tolerance. Those guards were confirmed to add coverage by mutating cpu/mha and observing every pre-existing test stay green while only the new guard failed.

RESEARCH CAPTURE: R-01KYM4HGM1EEY is superseded by R-01KYM8N2N2E56 and should be read only through it. R-01KYM6C5AWFKZ stands as corrected by R-01KYM78A9BEBC. The method rules extracted — PROC-009 through PROC-015 — are the durable output, and five of the seven exist because a probe of mine produced a wrong answer first.

## R-01KYMAABNFEXZ9NM6HRPXRKM0R Compliance audit of the session against existing spec rules: 3 violations, 1 clean, 1 false clean
kind: research
state: draft
created: 2026-07-28

Audited this session's own work against the repo's existing spec rules rather than against recollection. Three violations found and fixed, one rule verified clean, and the method is the transferable part: every gap was invisible from memory and obvious from a mechanical check.

VIOLATIONS FOUND AND FIXED
  PERF-014 (detector needs a positive AND negative fixture) — PS4005 shipped with NEITHER, while PS4006 and PS6001 complied. Found by counting fixtures per detector instead of recalling that all three had been tested. Fixed in 0992b2b with three fixtures, the load-bearing one being silence on the HOISTED odometer: PS4005 initially reported the sites it had just helped fix, so that fixture pins the discriminator rather than restating the rule.
  PERF-009 (record a benchmark AND its baseline) — benchmarks were added with each change and CI smoke-runs them, but baselines lived only in commit messages, which no later session searches. docs/perf-notes-*.md is where this repo keeps them and eighteen optimizations had not touched it. Fixed in 9269f90.
  PATTERNS.md completeness — the three new detectors were absent, and more consequentially the config section documented every tuning key EXCEPT shapeMethods, which ca174c2 introduced. An undocumented key in an otherwise complete reference is worse than no reference, since a reader concludes the list is exhaustive. Fixed in 4f408ae.

VERIFIED CLEAN
  ARCH-012 (a pure-Go ref kernel for every registered Op) — 91 ops declared in backend/op.go, 90 registered in backend/ref, and the single gap is OpInvalid, a sentinel. Satisfied.

A FALSE CLEAN, worth recording because it nearly passed: the first ARCH-012 check extracted 48 declared ops against 90 ref kernels and reported an empty gap set. An empty diff from a wrong input set looks exactly like compliance. The op list was being scraped from the wrong files; against backend/op.go the real numbers are 91 and 90. A compliance check needs its INPUT verified before its output is believed — the same discipline PROC-012 and PROC-013 apply to mutation probes and string edits.

WHY THIS BEATS SELF-REVIEW: all three violations concerned work done hours earlier in the same session, and in each case the recollection was that the requirement had been met. PERF-014 in particular was being satisfied by habit on two detectors out of three, which is exactly the pattern a memory-based review confirms rather than catches.

NOT YET AUDITED: NUM-011 (golden test against NumPy or PyTorch within documented tolerance), NUM-012 (finite-difference gradient check), NUM-016 (NaN/Inf, empty and zero-dim tensor policy). These apply per-operation rather than per-session and would be the natural next sweep for anyone continuing this line.

## R-01KYMADB2VE1PS3APR1XG6TXQE NUM-016 spot-check on the session's fast paths: no edge-case regression
kind: research
state: draft
created: 2026-07-28

Verified that this session's fast paths did not regress NUM-016 edge-case handling. Clean result — no action taken, and no test added.

WHAT WAS CHECKED: every op whose fast path was added or changed this session, driven with a zero-dimensional tensor, an empty tensor and a length-1 tensor, with panics captured rather than allowed to fail the run. OpSum zero-dim and empty, OpArgMax zero-dim and len-1, OpDot len-1, OpDPO and OpIPO len-1, OpCholesky 1x1. No panics. OpArgMax with Axis 0 on a rank-0 input returns an error, which is correct rather than a defect.

WHY IT MATTERED: the devirtualizations all added a new branch — a flat typed view guarded by f64Data, with the accessor loop kept as a fallback — and several added explicit rank-0 or single-element guards. A new branch that panics on an empty or zero-dim tensor is exactly the regression this class of change risks, and it would not be caught by the interleaved A/B benchmarks (which use well-formed inputs) nor by the bit-identity oracles (same).

INPUT VERIFIED before believing the empty result, per PROC-017: the probe uses F64 tensors, so f64Data succeeds and the FAST path is what executes, not the fallback. A probe that silently exercised the fallback would have proved nothing about the new code.

NO TEST ADDED, deliberately. NUM-016 already requires each operation to test its policy for NaN, Inf, empty, zero-dimensional and non-contiguous inputs with those cases in its golden file, so a session-scoped edge-case test would duplicate per-op coverage that the spec already mandates — the same redundancy that cost ten oracles and a conv parity test earlier today (PROC-014). If a specific op turns out to lack that coverage, the fix belongs in that op's golden file, not in a cross-cutting probe.

LIMIT OF THE CHECK, stated so it is not over-read: it verifies no panic and no crash, not edge-case VALUES. NaN and Inf propagation and non-contiguous views were not exercised. Those are per-op obligations under NUM-016 and remain unaudited, alongside NUM-011 and NUM-012.

## R-01KYMVK2M9E0ZB48GMHX6ADG04 Four optimization candidates declined on measurement — MLA, AQLM, SparseGPT, cholSolve
kind: research
state: draft
created: 2026-07-28

Each was implemented far enough to benchmark, then reverted. Recording the numbers so the next agent reading the same candidate does not re-derive them.

MLA (backend/ref) — 0.85x and 0.82x. The transform made it SLOWER on both shapes tried. Not marginal, not noise; the rewrite loses more than it gains and the direction is settled.

AQLM (nn) — 0.9958x. A null. Inside noise, and on the wrong side of 1.0.

SparseGPT (nn) — 1.0046x. A null, and the methodologically important one. A first, NON-INTERLEAVED run reported 1.048x and was published before correction; separate baseline and after runs had measured machine drift and dressed it as a 4.8% win. Interleaved in one session it collapses to 1.0046x. Three published figures had to be corrected off the back of this. It is the reason PROC-INTERLEAVE-001 is not optional: a separate-runs A/B does not report a slower or faster code path, it reports whatever the machine was doing at the time.

cholSolve (autograd) — 0.93x. Slower. Distinct from the Cholesky VJP work that DID ship at 1.025x; the solve itself declined.

RELATED MEASUREMENT HYGIENE from the same campaign, worth carrying: one Cholesky measurement at n=64 was thrown out as unusable rather than reported — the OLD arm swung 87% within a single set and would have read as 17% SLOWER. Re-run at n=128 it was stable. An arm that will not hold still is not a result, in either direction.

STANDING: none of these four is suppressed in perfscan. They are declined at the measured sizes on this host (Apple M2 Pro, darwin/arm64, go1.26.5). A different shape or a machine with different memory behavior could move them, but the burden is a fresh interleaved measurement, not an argument from the code shape.

## ADR-01KZ3HW0ZSFE7T23XD65GPBRE6 Lower classic treeRadixCutoff from 512 to 32? It is worth 17.3 percent on the forest fit and changes which trees are grown.
kind: adr
state: done
created: 2026-08-03
context: MEASURED, interleaved over three rounds: BenchmarkForestFit 123.3 to 101.9 ms, minus 17.3 percent, and the variance disappears - 101.9 ms every run at 32 against 123 to 144 ms at 512. TreeFit is flat. The comparison-sort path is about half the forest fit: radixByFeature is 50.7 percent of a parallel profile and its closure comparator, the pdqsort partition and the insertion sort are most of that. Cutoffs of 32 and 128 both land near 102 ms. THE COST: the bit-exact forest digest FAILS at 32, 64 and 128 and passes only at 512, so the change alters which trees are grown. WHY, AND THIS CORRECTS A CLAIM IN THE CODE: radixByFeature documented its unspecified tie order as irrelevant because thresholds sit between distinct values. That is wrong - the sweep skips a candidate cut when the gap between consecutive values is at most featureThreshold, a TOLERANCE of 1e-7, so values that are distinct yet closer than that behave like ties, and reordering them changes which pairs are adjacent and therefore which cuts are considered. Two sorts that disagree on those runs grow different trees. The comment has been corrected; no behavior was changed.
decision: Keep 512 - preserve the exact trees the frozen digest pins
consequences: THE QUESTION WAS MALFORMED AND THE ANSWER IS BOTH OPTIONS AT ONCE. treeRadixCutoff was serving two callers with opposite cost profiles: a PER-NODE sort of a shrinking range in the CART builder, which wants a low cutoff, and a ONE-TIME presort of every row in the GBM builder, which wants a high one. Splitting it resolves the tradeoff instead of choosing a side. The GBM presort keeps 512 as the new gbmRadixCutoff - that is the option chosen here - and the CART per-node cutoff drops to 32. With the two separated: BenchmarkForestFit 121.5 to 101.3 ms, minus 16.6 percent with all three new runs below all three base runs; GBMFit flat at 66.5 to 64.0; and BOTH bit-exact digests pass UNCHANGED. The apparent model-behavior change was entirely the shared constant dragging the GBM presort along with the CART one, which also cost GBMFit about 25 percent. No trees change and nothing is re-frozen.
status: accepted

kind: radio
option: Keep 512 - preserve the exact trees the frozen digest pins
option: Lower to 32 and re-freeze the digest - accept different but equally valid trees for 17.3 percent
option: Lower only for classification, where the risk is smallest, and keep 512 for regression
choice: Keep 512 - preserve the exact trees the frozen digest pins

## R-01M01CYGEBETMARNK7BDJ7P8S3 M2 Metal prefill quant matmul re-reads the weight per row: 87x at M=256, 0.31 TFLOP/s effective
kind: research
state: draft
created: 2026-08-15

MEASURED, and it is the largest gap found on this device so far. Benchmark shipped on the perf/m2-metal-kquant-cooperative branch (backend/metal/q4k_msweep_bench_test.go); no production code changed.

WHERE IT COMES FROM. The M2 Metal cooperative K-quant kernels (T989/T993, 1.69-11.79x) are selected by `cooperative = M == 1 && enabled` in metal_bridge.m. That gate is correct for what it covers - at M=1 only N threads have work and each walks all of K, which is the occupancy problem the cooperative kernel solves. But it means every M>1 dispatch, which is ALL prefill and batched decode, runs the scalar kernel. Nothing had measured that path.

THE MEASUREMENT ISOLATES THE QUESTION BY CONSTRUCTION: weight bytes do not depend on M, so ns/op alone distinguishes a kernel that reads each weight once and reuses it across M rows from one that re-walks the weight per row. K2048,N5632 (TinyLlama SwiGLU gate/up), 100x count=2:
     M     ns/op   vs M=1   useful GB/s   actual traffic GB/s
     1    221428     1.0x          29.3                  29.3
     2    439192     2.0x          14.8                  29.5
     4    682965     3.1x           9.5                  38.0
     8    937278     4.2x           6.9                  55.4
    16   1468692     6.6x           4.4                  70.7
    32   2727207    12.3x           2.4                  76.1
    64   5032750    22.7x           1.3                  82.5
   128   9799232    44.3x           0.7                  84.7
   256  19318708    87.2x           0.3                  86.0

LINEAR IN M from M=8 onward, about 1.9x per doubling, while the weight is 6.49 MB at every M. So each of the M rows re-reads and re-dequantizes the whole weight matrix. Useful weight throughput falls 29.3 -> 0.3 GB/s across the sweep.

THE KERNEL IS NOT BADLY WRITTEN - actual traffic saturates near 86 GB/s, a respectable fraction of this machine's bandwidth. It is the WRONG SHAPE: a matvec used as a matmul, moving M times the data the problem requires. That distinction matters for the fix, which is a different kernel, not a tuning pass on this one.

HEADROOM, stated as the measured floor rather than a predicted speedup: at M=256 the dispatch is 5.91 GFLOP in 19.32 ms = 0.31 TFLOP/s effective, on a GPU with several TFLOP/s of f32. Reading the weight once would make the weight traffic 6.49 MB instead of 1.66 GB. The realistic target is a compute-bound kernel; the exact multiple is NOT claimed here because no tiled kernel has been built or measured yet, and this repo's own history is full of predicted speedups that did not survive an interleaved A/B.

THE FIX SHAPE, from the same source already used for the M=1 kernel: llama.cpp keeps separate mat-vec and mat-mat quant kernels for exactly this reason. Stage a dequantized weight block once into threadgroup memory, reuse it across the M rows of the tile, accumulate per row. The existing cooperative work established the capability gate, the forced-off control and the paired-alternating measurement protocol, so all three carry over.

WHY THIS OUTRANKS THE REMAINING BACKLOG. Three consecutive detector-derived candidates measured out at zero or near-zero on this machine (crossentropy math.Log inapplicable, FA /l norm 4.1 percent on one dtype, Attrs boxing 0.8 percent allocs and no time). This one is 87x of redundant work at a realistic prefill batch, found by asking what the hardware is not being used for rather than by triaging detector output. Prompt processing is half the serving story and it is the untouched half.

NEXT STEP IS AN IMPLEMENTATION TASK, not more analysis: a tiled Q4_K mat-mat kernel behind the same capability gate, forced-off control, bit-identity against the scalar kernel, and the interleaved A/B with an unaffected control. Q6_K follows the same shape once Q4_K lands.

## R-01M01DD2A5ER9V02174WE7XXKT REJECTED M2 Metal Q4_K M-blocked mat-mat: 2.2-6.3x slower, occupancy beats traffic; corrects the 87x framing
kind: research
state: draft
created: 2026-08-15

REJECTED BY MEASUREMENT, and it corrects an overstatement in R-01M01CYGEBETM. Candidate code removed; backend/metal is byte-identical to the branch state before the experiment.

WHAT WAS BUILT: qmatmul_q4k_mrows, an M-blocked variant of the resident Q4_K kernel. The scalar kernel gives one thread per output element, so the M threads sharing an output column each re-walk and re-dequantize the same weight row. The candidate gave one thread a column for a block of MT=8 rows, dequantizing each weight once into 8 accumulators - cutting weight traffic 8x. Full plumbing: pipeline, capability flag, SetQ4KMRows forced-off control, dispatch selection for M>1.

CORRECTNESS WAS ESTABLISHED FIRST AND IT HELD. Bit-identical to the scalar kernel across M in {2,7,8,9,17,64}, covering partial blocks (7), exact blocks (8), remainders (9,17) and many blocks (64) - the accumulation order per output element is unchanged, so this was bit-identity by construction and it verified. Non-vacuity proven by mutation: scaling the accumulate by 1.0001 turned the test red in 256 of 256 elements, and reverting returned it green, so the kernel genuinely engaged rather than the test comparing scalar against scalar.

THE PERFORMANCE RESULT, interleaved same-binary A/B via SetQ4KMRows, K2048,N5632, 100x count=2, medians:
     M   mrows ns   scalar ns   mrows/scalar
     2    2775442      438952          6.32x SLOWER
     4    3452463      703100          4.91x
     8    5157598      935674          5.51x
    16    5224976     1469328          3.56x
    32    9728132     2808948          3.46x
    64   13104634     5181642          2.53x
   128   22101794     9971549          2.22x
   256   41855208    19339916          2.16x
Slower at EVERY M, by 2.2x to 6.3x. The hypothesis is refuted, not merely unconfirmed.

WHY: the candidate traded 8x fewer weight reads for 8x fewer threads. At M=256 the scalar kernel launches M*N = 1441792 threads; mrows launches N*ceil(M/8) = 180224. The GPU was not weight-bandwidth-bound - it was occupancy-bound - so removing parallelism cost far more than removing traffic bought. Per-thread register pressure from 8 accumulators and the K-strided X reads (stride 2048 floats between accumulators) compound it.

THE CORRECTION TO R-01M01CYGEBETM, which matters more than the failed candidate. That record framed the M-sweep as "87x of redundant work" and implied large headroom. The arithmetic there was misread: at M=256 the kernel performs 256x MORE arithmetic than at M=1 (2.95 G MACs against 11.5 M), while time grew only 87x. Efficiency per MAC therefore IMPROVES with M - the scalar kernel amortizes its launch and latency better as the batch grows, which is the opposite of the deterioration the record suggested. The weight re-reading is real, but it is not the binding constraint, and this experiment is the proof.

WHAT REMAINS OPEN, honestly narrowed: the only surviving signal is 0.31 TFLOP/s effective at M=256 against a GPU with several TFLOP/s of f32. That is a compute-efficiency gap, not the traffic gap the earlier record described. The untested approach is threadgroup-memory tiling: a threadgroup cooperatively dequantizes a weight tile into shared memory and MANY threads covering many M rows reuse it, so weight reads drop WITHOUT dropping thread count. That is the one shape this experiment did not test and the only one whose failure mode is not already demonstrated. It is materially harder than the register-blocking tried here.

METHOD NOTE: the correctness work was completed before the benchmark and cost nothing when the candidate was rejected - bit-identity, block-boundary shapes and the mutation probe all held. What failed was the performance hypothesis alone. Building the forced-off control first is what made the A/B a same-binary comparison and the refutation unambiguous.

## R-01M01DJ74WF738NXP1A03K10MR M2 Q4_K prefill roofline: dequant ALU worth at most 1.41x, perfect weight reuse at most 1.67x
kind: research
state: draft
created: 2026-08-15

CEILING ESTABLISHED BY TWO CHEAP PROBES, after one expensive kernel failure. Both probes reverted; backend/metal is byte-identical to the branch. This closes the M2 prefill-quant-matmul arc with a bound rather than another candidate.

WHY PROBE BEFORE BUILDING: R-01M01DD2A5ER9 rejected an M-blocked kernel that was correct, bit-identical and 2.2-6.3x SLOWER, because it traded threads for traffic and the kernel turned out to be occupancy-sensitive. That cost a full implementation to learn one bit of information. These two probes cost one line each and bound BOTH remaining levers.

PROBE A - dequantization ALU. Keep every weight-byte load, the grid shape and the thread count; drop only the nibble select and scale reconstruction (acc += X*(float)qbyte instead of X*(dl*nib - ml)). This isolates the unpacking arithmetic.
PROBE B - weight traffic. Keep the ALU, the grid and the thread count; set rowOff=0 so every thread reads the SAME weight row. Identical instruction and load COUNT, but the whole working set is 1152 bytes and stays L1-resident, i.e. perfect weight reuse. Outputs are wrong by construction; only the timing is read.

MEASURED, K2048,N5632, 100x count=2, medians:
     M   baseline   A no-dequant      B perfect-reuse
     8     935674   734084  1.27x     707478  1.32x
    64    5181642  3600288  1.44x    3134630  1.65x
   256   19339916 13724966  1.41x   11570676  1.67x

READING. Removing ALL dequantization arithmetic buys 1.41x at M=256. Giving the kernel PERFECT weight-cache behavior buys 1.67x. Those are ceilings, not estimates: no real kernel can beat free ALU or free memory. Any weight-sharing scheme - threadgroup-memory tiling included - targets the second lever and therefore cannot exceed 1.67x, and it must pay tiling overhead, barriers and the occupancy risk that already killed the register-blocked attempt.

CONSEQUENCE FOR THE OPEN CANDIDATE. Threadgroup tiling is still the only shape whose failure mode is not already demonstrated, but its prize is now known to be at most 1.67x and realistically well under that. It is no longer justified by "prefill is 87x redundant" - that framing was wrong and has been corrected in R-01M01DD2A5ER9. Whether a hard shared-memory kernel is worth at most 1.67x on the prefill half of serving is a scoping decision, not a technical unknown.

WHAT THE NUMBERS ALSO SAY ABOUT THE KERNEL. Neither lever dominates: 29 percent of time is dequant ALU, and the memory side yields only 40 percent even when made free. Nothing is saturated - not bandwidth (86 GB/s measured against roughly 200 available), not compute, not dequant. That signature is latency-bound execution with enough parallelism to partly hide it, which is consistent with the register-blocked kernel collapsing the moment thread count dropped 8x. It also means the kernel is closer to reasonable than the earlier record implied.

METHOD, worth carrying: to bound a lever, disable it in place while holding grid shape, thread count and instruction count fixed, and read only the time. Probe A holds memory constant and removes ALU; probe B holds ALU constant and removes memory cost. Two one-line edits bounded an optimization space that a full implementation had failed to bound.

## R-01M01EX9BCE6SVV6A3ZF90WF3K Three K-quant types (Q2_K/Q3_K/Q5_K) still scalar at M=1 while Q4_K/Q6_K run cooperative; Q5_K is next
kind: research
state: draft
created: 2026-08-15

MEASURED OPPORTUNITY, benchmark shipped on perf/m2-metal-kquant-cooperative (backend/metal/kquant_gap_bench_test.go). No production code changed.

THE STRUCTURAL POINT: the simdgroup-cooperative M=1 kernels cover Q4_K and Q6_K only. Q2_K, Q3_K and Q5_K still dispatch the scalar one-thread-per-output-row kernel. At M=1 that kernel gives work to only N threads and each walks all of K, which is precisely the occupancy shape the cooperative work was built to fix. The shape is a property of the dispatch, not of the quant format, so it applies to all five types.

MEASURED at K2048,N2048, 300x count=3, medians with the warmup sample dropped. Block sizes differ (84/110/144/176/210 bytes per 256 weights), so the weight-byte rate is the comparable figure:
  Q3_K  scalar only        962 MB/s
  Q2_K  scalar only       1182 MB/s
  Q5_K  scalar only       1896 MB/s
  Q4_K  COOPERATIVE       7649 MB/s
  Q6_K  COOPERATIVE      12754 MB/s
Q4_K's own scalar kernel runs about 3010 MB/s at this shape (2.36 MB in 783942 ns, from the interleaved re-verification of PR 1061). So the three uncovered types sit at or below scalar-class throughput while the two covered types run several times higher.

WHAT THIS DOES AND DOES NOT ESTABLISH. It is NOT a like-for-like efficiency claim across formats: Q2_K and Q3_K unpack more bits per byte than Q4_K, so part of the spread is inherent. The defensible reading is narrower and sufficient - the three uncovered types are at scalar-class throughput, and scalar-to-cooperative on this exact code path is a MEASURED 3.41x median (2.43x on the most conservative reading, best scalar against worst cooperative, non-overlapping distributions).

WHY THIS CANDIDATE IS DIFFERENT FROM THE ONE THAT FAILED. R-01M01DD2A5ER9 rejected an M-blocked mat-mat kernel because its hypothesis - that weight traffic bound the kernel - was wrong, and the roofline in R-01M01DJ74WF738NXP1A03K10MR later bounded that whole lever at 1.67x. Here the hypothesis is not a hypothesis: the same transform has already been applied twice on the same dispatch path and measured 3.41x (Q4_K) and 2.69-11.79x (Q6_K). The remaining work is porting a proven kernel to three more block layouts, not testing an idea.

BUILD ORDER: Q5_K first. It has the highest rate of the three, so its result is least confounded by unpack complexity, and Q5_K_M is among the most widely used GGUF quantizations in circulation. Its 5-bit format splits each weight across a 4-bit qs nibble and a qh high-bit plane, which is the one real structural difference from the Q4_K kernel it would be adapted from. Q3_K and Q2_K follow; both carry more complex scale/min packing and should be judged on their own A/B rather than on Q5_K's result.

REQUIRED PER TYPE, following what PR 1061 already established: capability gate, forced-off Set*Cooperative control, bit-identity or documented-tolerance test against the scalar kernel across block-boundary shapes, mutation probe to prove the test is not comparing scalar against scalar, and an interleaved warmup-trimmed A/B per FIRST-BENCHMARK-SAMPLE-IS-NOT-COMPARABLE-001.

## R-01M01KBSTZFY5THEBS6DGFMYF9 GPU quant decode M=1 campaign closed: 14 cooperative kernels, 7 formats, 2 backends, 1.80x-6.01x
kind: research
state: draft
created: 2026-08-15

CAMPAIGN CLOSED. Every quantized matmul format on both GPU backends now has a cooperative M=1 kernel. 14 kernels, 7 formats, 2 backends, each with parity across five shapes, targeted mutation probes and an interleaved warmup-trimmed A/B.

  format   Metal            Vulkan
  Q2_K     2.21x            2.19x
  Q3_K     6.01x            2.54x
  Q4_K     3.41x            2.17x
  Q5_K     2.66x            3.04x
  Q6_K     2.69-11.79x      2.20x
  Q4_0     2.48x            1.80x
  Q8_0     3.02x            1.88x

THE DEFECT was structural and identical on both backends: the scalar kernels dispatch ONE THREAD PER OUTPUT ELEMENT, so at M=1 only N threads have work and each walks all of K. The Vulkan Q4_K shader stated it in its own header ("One invocation per output (mi,ni)") long before anyone measured what it cost. Filed as ONE-THREAD-PER-OUTPUT-IS-AN-M1-OCCUPANCY-DEFECT-001.

THE FIX is one simdgroup (Metal, 32 lanes) or workgroup (Vulkan, 64 invocations) per output row, splitting that row's K and reducing with simd_sum or a shared-memory tree. The split must follow the FORMAT'S SCALE GROUPS rather than cut across them; where it does, per-element arithmetic stays byte-for-byte identical to the scalar kernel and only the summation order changes.

THREE SPLIT SHAPES COVERED IT ALL. (1) 8 groups of 32 (Q4_K, Q5_K): Metal 8 elements per lane, Vulkan 4 per invocation. (2) 16 groups of 16 (Q2_K, Q3_K): Metal 2 lanes per group, Vulkan 4 invocations per group. (3) Q6_K needed none — its scalar body is already 2x32 iterations each producing 4 outputs, so 64 invocations is exactly one iteration each. The 32-weight legacy formats (Q4_0, Q8_0) could not subdivide a block at all and SPAN blocks instead: element I%32 of every block with parity I/32.

PREDICTIVE RESULT, and it held across all 14: THE GAIN SHRINKS AS DEQUANT GETS CHEAPER. Q3_K and Q2_K do the most bit-unpacking per byte and gained most on Metal; Q8_0 and Q4_0 are nearly free to dequantize (one int8 multiply, one nibble plus an offset) and gained least on both backends. Simple formats are therefore worth doing LAST, not first — the opposite of the intuition that simpler kernels are easier wins.

TOLERANCE DIFFERS BY BACKEND AND THAT MATTERED. Metal used a flat 2e-5 relative bar, inherited from the existing Q4_K/Q6_K parity tests. Copying that bar to Vulkan FAILED at k=2048,n=64 (2.640e-05) — not a kernel defect but cancellation: those shapes produce a small result from large cancelling partial sums, where reassociation is amplified. The right bar is the Vulkan package's own crossTol(k) = 2.5e-6*sqrt(k), which scales with K exactly as f32 accumulation error does.

WHAT THE MUTATION PROBES CAUGHT, because this is the transferable part. A Metal Q8_0 parity test PASSED with max difference 0.000e+00 across all five shapes — the signature of comparing scalar against scalar. Perturbing the kernel by 50 percent left it GREEN: QMatMulQ8_0 reaches a standalone entry point and only the RESIDENT dispatch had been wired. An exactly-zero difference looked like the best possible result and was the worst. Every later kernel probed both dispatch paths from the start. Derived shaders got probes aimed specifically at what distinguishes them from the sibling they were derived from (Q5_K's qh plane, Q2_K's packed scale/min nibbles, Q4_0's x[i]/x[i+16] interleave, Q3_K's rewritten single-scale selection), because everything inherited is already correct and only the difference can be wrong.

REJECTED ALONG THE WAY, both recorded: an M-blocked mat-mat kernel for prefill (2.2-6.3x SLOWER — it traded 8x fewer weight reads for 8x fewer threads, and the kernel is occupancy-bound not bandwidth-bound), and the prefill lever generally, bounded by roofline probes at 1.41x for dequant ALU and 1.67x for perfect weight reuse.

NEXT: the same defect shape should be checked on CUDA, which is the parallel worker's lane and was not touched here.

## R-01M01M2PETFG2SCMJVBCK9SNG7 REJECTED Vulkan M=1 GEMV: 3.9 percent slower — the wasted tile threads were buying occupancy
kind: research
state: draft
created: 2026-08-15

REJECTED BY MEASUREMENT, and it corrects R-01M01... the f32 GEMM class-audit finding from the previous iteration. Candidate code removed; backend/vulkan is byte-identical apart from the probe benchmark, which stays.

WHAT WAS BUILT: matmul_gemv.comp, an M=1 specialization of the tiled f32 GEMM. matmul.comp computes a 16x16 output tile with a 16x16 workgroup, so at M=1 only the ty==0 row has real data and 15/16 of the arithmetic is discarded. The specialization gave each thread one output column walking K, with coalesced b[k*N+col] reads, plus full plumbing: vk_recorder_gemv with its own ceil(N/256) dispatch, SetGEMV control, embedded SPIR-V.

CORRECTNESS WAS PERFECT AND IS NOT THE POINT. Bit-identical to the tiled kernel — 0.000e+00 across all five shapes — because both accumulate in ascending k with the same additions in the same order. That exact-zero is the signature that was VACUOUS on Metal Q8_0, so it was probed rather than trusted: perturbing the accumulate turns the test red at 9.981e-04 and transposing the B index at 2.230e+01, so the kernel genuinely ran and the bit-identity is real.

MEASURED, interleaved A/B via SetGEMV, six alternations, -benchtime=200x -count=2, K2048,N2048:
  gemv  median 1088170 ns   range  823489-1517943
  tiled median 1047505 ns   range  893321-1093654
The specialization is 3.9 percent SLOWER, with heavily overlapping distributions. Eliminating 15/16 of the arithmetic bought nothing.

WHY, and this is the correction. The previous record called the M=1 case "tile-waste-bound" and read 15.3 GB/s against ~200 GB/s as headroom. Wrong diagnosis. Thread counts at M=1, K=N=2048:
  tiled: ceil(N/16) x ceil(M/16) = 128 workgroups x 256 = 32768 threads
  gemv : ceil(N/256)             =   8 workgroups x 256 =  2048 threads
The specialization has SIXTEEN TIMES FEWER threads in flight. The tiled kernel's "wasted" 15/16 of threads are exactly what supplies the occupancy that hides memory latency; removing the waste removes the latency hiding with it. Neither kernel saturates bandwidth because both are latency-bound, not bandwidth-bound.

SECOND CONFIRMATION OF THE SAME PRINCIPLE. R-01M01DD2A5ER9 rejected a Metal M-blocked mat-mat kernel for the identical reason: it traded 8x fewer weight reads for 8x fewer threads and came out 2.2-6.3x slower. Two independent attempts, two backends, same failure mode — on these GPUs, at these shapes, REDUNDANT WORK IS CHEAPER THAN LOST PARALLELISM. That is the opposite of the intuition that drove both attempts, and it is now the prior for any future "eliminate the wasted work" proposal here.

WHAT THE COOPERATIVE QUANT KERNELS DID DIFFERENTLY, and why they won where these lost: they INCREASED thread count. One thread per output row became one simdgroup or workgroup per output row — 32x or 64x MORE threads, not fewer. The 14 successful kernels and the 2 rejected ones separate cleanly on that axis, not on how much redundant work they removed.

STILL UNTESTED: a split-K GEMV that keeps thread count high by giving each output column several workgroups and combining them, which needs atomics or a second pass. It is the only shape not refuted, and the two failures above are reason to bound it by probe before building it.

## R-01M01MCSPZF73A3GMXHSAW6R5T The 149us submit floor is a benchmark artifact: production batches a whole decode step per submit
kind: research
state: draft
created: 2026-08-15

QUESTION CLOSED, and it corrects the recommendation in the previous record. That one measured a ~149us per-submit floor and concluded "the actionable item is batching, not the kernel". Production ALREADY batches. The floor is a benchmark artifact, not an inference cost.

WHAT PRODUCTION DOES. llamagpu/decoder.go creates one recorder at line 3010 and finishes at 3342, with the whole per-layer loop (`for _, b := range d.blocks`) inside that scope — every layer's ops record into the same command buffer. Vulkan's Recorder.Commit() is a NO-OP (`return nil`) and Wait() calls Finish(), so the actual submits are the Wait partway through and the final Finish: roughly TWO submits per decode step, each carrying all the layer ops recorded up to it. gpt.go (127->159, 211->254) and t5_decoder.go follow the same one-recorder-per-step shape.

SO THE FLOOR IS AMORTIZED IN PRODUCTION. Two submits per token at ~149us is ~0.3ms. Against the 16.35 tok/s (61ms/token) the Metal Q4_K+Q6_K work reached, that is about 0.49 percent of a decode step. Even at a pessimistic 5 tok/s it stays around 0.15 percent. There is no batching problem to fix.

WHAT REMAINS TRUE FROM THE PREVIOUS RECORD: every benchmark in this series creates a recorder, records ONE op, finishes and frees per iteration, so all of them paid the floor on BOTH sides of their A/B. The comparisons are valid (common-mode) but the ratios are DILUTED, and the reported GPU speedups are LOWER BOUNDS on kernel improvement — Metal Q4_K cooperative reads 3.41x as measured and 7.85x with the floor removed from both sides.

WHAT WAS WRONG: the inference that a large per-submit cost implies a batching opportunity. It does only if the caller submits per op, and this caller does not. The benchmark harness submits per op because that is how you isolate one kernel; production does not because that would be absurd. I read a property of the measuring instrument as a property of the system.

NET EFFECT ON THE CAMPAIGN: the kernel work was the right target after all. The 14 cooperative kernels operate on the part of the decode step that is NOT submit overhead — roughly 99.5 percent of it — and their true improvement is larger than reported, not smaller.

STANDING CHECK worth keeping: before treating a fixed per-call cost as an optimization target, find the production call site and count how many real operations it amortizes that cost across. The harness's call pattern is chosen for isolation and is not evidence about the system.

## R-01M01N2NFAEJAVQ3JVP12JJN1C End-to-end: Metal 2.38x/2.63x, Vulkan 1.39x — and seven Vulkan shaders were unreachable from the decoder
kind: research
state: draft
created: 2026-08-15

END-TO-END VALIDATION OF THE WHOLE CAMPAIGN, and it found the campaign's worst defect.

MEASURED, real Generate loop, TinyLlama-shaped projections (Dim 2048, Hidden 5632, 6 layers, 32 tokens), cooperative toggled through the exported setters, three alternations, medians:
  METAL   Q4_K   52.75 -> 125.61 tok/s = 2.381x   (off 52.7-52.9, on 124.8-128.2)
          Q8_0   36.72 ->  96.69 tok/s = 2.633x   (off 36.7-37.2, on 96.0-97.4)
  VULKAN  Q4_K   50.95 ->  71.00 tok/s = 1.393x   (off 50.7-55.4, on 69.7-72.1)
          Q8_0   47.51 ->  65.78 tok/s = 1.385x   (off 46.8-48.8, on 64.4-65.8)
No overlap in any of the four.

THE DEFECT THIS FOUND. All seven VULKAN cooperative shaders were unreachable from the decoder. Cooperative selection had been wired into the standalone QMatMul entry points — which the parity tests and leaf benchmarks drive — but NOT into Recorder.QMatMulResident, which is what llamagpu calls. residentSpirv always returned the scalar shader. The first end-to-end A/B read 1.000x and 1.015x: both arms ran identical code, the same signature as a vacuous parity test. Seven leaf A/Bs, seven parity tests and twenty mutation probes had all passed.

METAL WAS VERIFIED, NOT ASSUMED. Its resident and recorder entry points share one switch(qtype) block, so wiring cooperative once covered both; Vulkan's two paths have different code and only one was wired. Reading the source suggested Metal was fine; the 2.381x/2.633x measurement proves it.

MODEL SHAPE DECIDES WHETHER A LEAF WIN SURVIVES TO THE TOP. The same test at a 124M-class config (Dim 768, Hidden 2048) reads about 1.04x even with everything wired, because those matmuls are too small for the kernel to dominate a token. A null result at small shapes is not evidence the kernel is worthless.

THE END-TO-END NUMBER IS SMALLER THAN THE LEAF NUMBER AND THAT IS CORRECT. Metal Q4_K measured 3.41x on the isolated leaf and 2.381x end to end; Vulkan Q4_K 2.17x and 1.393x. A decode step also does LayerNorm, RoPE, attention, residuals and host work the kernels do not touch. The end-to-end figure is the one worth quoting.

STANDING LESSON: a leaf benchmark proves a kernel is faster; only an end-to-end measurement through the PRODUCTION entry point proves the system calls it. Every correctness and performance gate in this campaign passed while the Vulkan feature was dead code. An attempt to file this as an EARS rule was rejected by the spec server's slot validation four times; recorded here instead so it is not lost.

## R-01M01NFK23EM09B17X2TMHB4FB Q2_K and Q3_K move ~40-50 percent of Q4_K's weight bytes per second — unpacking-bound, probe before building
kind: research
state: draft
created: 2026-08-15

MEASURED OBSERVATION, NOT A PROPOSAL. Normalizing the seven end-to-end Metal results by weight bytes shows two formats far off the pace, and names the next candidate without committing to a mechanism.

Metal cooperative, end to end, weight-bytes moved per second relative to Q4_K:
  fmt    tok/s   B/256w   relative
  Q8_0   78.14      272      1.36
  Q4_K  108.53      144      1.00
  Q6_K   73.52      210      0.99
  Q4_0   90.61      144      0.83
  Q5_K   71.81      176      0.81
  Q2_K   92.69       84      0.50
  Q3_K   59.84      110      0.42

Q2_K and Q3_K move roughly 40-50 percent of Q4_K's weight bytes per second DESPITE HAVING SMALLER BLOCKS. Every other format sits between 0.81 and 1.36, which is the spread bandwidth alone would explain. Those two are bound by something else — most plausibly the unpacking, since they are the two formats with the most bit manipulation per weight: Q3_K reads a separate hmask plane in addition to its 2-bit qs field, and Q2_K extracts four 2-bit fields from each qs byte under four different shifts.

WHY NO FIX IS PROPOSED HERE. The obvious mechanism — each qs byte is fetched once per shift, so four invocations touch the same byte — probably describes an L1 hit rather than a memory fetch, and the obvious remedy (one invocation handling all four shifts) would CUT invocation count fourfold. That runs straight into REDUNDANT-GPU-WORK-IS-CHEAPER-THAN-LOST-PARALLELISM-001, which was earned by two failures on exactly that trade: the Metal M-blocked mat-mat (2.2-6.3x slower) and the Vulkan M=1 GEMV (3.9 percent slower). Every unbounded "obvious fix" attempted in this campaign has failed; the ones that worked were bounded by probe first.

WHAT A PROBE WOULD LOOK LIKE, by analogy to the roofline pair that bounded the prefill lever at 1.41x and 1.67x: hold grid shape, thread count and instruction count fixed and disable one lever at a time. Replace the qs/hmask extraction with a constant while keeping every load, to bound the ALU share; and point every invocation at superblock 0 while keeping the arithmetic, to bound the traffic share. Two one-line edits, and they bound the space before any kernel is written.

STANDING VALUE EITHER WAY: even unimproved, Q2_K and Q3_K already gained 1.913x and 3.876x end to end from the cooperative work. This is about whether a second, smaller lever exists on top, not about a defect.

## R-01M01NJX03EV6AH33FCDGTZQYB REFUTED: Q2_K/Q3_K are not unpacking-bound — removing all unpack ALU is worth at most 1.08x
kind: research
state: draft
created: 2026-08-15

CANDIDATE KILLED BY PROBE, cost two one-line edits. Closes the question the previous record opened. Tree restored; nothing shipped.

THE HYPOTHESIS. Normalized end-to-end results showed Q2_K and Q3_K moving 0.50x and 0.42x the weight bytes per second that Q4_K does, despite SMALLER blocks, while every other format sat between 0.81x and 1.36x. The proposed explanation was the unpacking: those two do the most bit manipulation per weight — Q3_K reads a separate hmask plane on top of its 2-bit qs field, Q2_K extracts four 2-bit fields per qs byte under four shifts.

THE PROBE held grid shape, thread count and LOAD COUNT fixed and removed only the unpacking arithmetic: the shift-and-mask on qs and the hmask bit test became plain byte reads, so every memory access still happens and only the ALU work disappears. Interleaved A/B, six alternations, -benchtime=300x -count=2, first sample of each run discarded:
  baseline (full unpack) median 342538 ns   range 332121-616673
  no-unpack-ALU          median 317383 ns   range 285583-338452
  ceiling 1.08x, distributions OVERLAPPING.

REFUTED. Removing ALL of Q3_K's unpacking arithmetic is worth at most 8 percent and cannot be distinguished from noise. The unpacking is not the bottleneck, so no kernel targeting it can be either. Q2_K shares the structure and is not probed separately on that basis.

WHAT THE DIFFERENCE PROBABLY IS, stated as unverified: load ISSUE count rather than bytes or arithmetic. Q3_K touches two byte planes per element while Q4_K reads one byte per two elements, so Q3_K issues roughly four times the loads per weight. That is a property of the format, not of the kernel, and there is no version of the kernel that reads fewer bytes than the format stores.

NET: no actionable lever here. Q2_K and Q3_K already gained 1.913x and 3.876x end to end from the cooperative work, and the residual gap to Q4_K is inherent to what those formats require a reader to do.

METHOD NOTE. The first attempt at this probe ran the two variants as SEPARATE blocks rather than interleaved and produced 1.14x with a 703005 outlier — unusable, and in the direction that would have justified building something. Re-running it interleaved with warmup trimming gave 1.08x with overlap. The discipline that produced the usable number is the same one filed as FIRST-BENCHMARK-SAMPLE-IS-NOT-COMPARABLE-001, and I violated it before following it.

## R-01M01P013VEVWSP463ZWJMH7AA The 7.1x llama.cpp gap profiled: 8 percent of peak bandwidth vs 57 percent, with batching already optimal
kind: research
state: draft
created: 2026-08-15

PROFILED. The 7.14x gap to llama.cpp is entirely kernel bandwidth efficiency. Submit batching, host CPU and dispatch count are all ruled out by measurement. Instrumentation reverted; nothing shipped.

THE NUMBERS, TinyLlama-1.1B Q4_K_M (636.18 MiB = 667.1 MB of weights re-read per token) on this M2 Pro:
  llama.cpp 48d22e295   172.19 tok/s    5.81 ms/token   114.9 GB/s   57.4 percent of the 200 GB/s peak
  GoAI                   24.11 tok/s   41.48 ms/token    16.1 GB/s    8.0 percent of peak
The ratio of achieved bandwidths is 7.13x and the ratio of token rates is 7.14x. The gap IS the bandwidth; there is nothing else to explain.

RULED OUT — HOST CPU. A Go CPU profile over the decode captures 3.28s of samples across 10.37s wall, i.e. 0.32 cores busy, and 53.96 percent of that is runtime.cgocall with 39.33 percent unknown (driver). The host is idle waiting on the GPU, not computing.

RULED OUT — SUBMIT OVERHEAD, and this contradicted my own prior. A ~149us per-submit floor was measured earlier in this campaign, and 41.48 ms/token divided by roughly 154 quant matmuls gives 269 us each, which is suspiciously close to that floor plus a kernel. Counting the actual submit primitives settles it: for 16 generated tokens the decoder issues Commit=16, Wait=16, Finish=1 — EXACTLY ONE COMMAND BUFFER PER TOKEN, carrying all 22 layers. Batching is already optimal and cannot be the lever.

WHAT THAT LEAVES. GoAI's Metal kernels move 16.1 GB/s where llama.cpp's move 114.9 GB/s on the same weights on the same device, inside an equally well-batched command buffer. The cooperative work in this campaign moved this model from 7.00 to 24.11 tok/s (3.44x) and did so by fixing occupancy; what remains is a different axis — how efficiently each thread reads memory once it has work.

CANDIDATE, EXPLICITLY UNPROBED. The kernels I wrote for Q2_K/Q3_K/Q5_K index the weight as individual bytes (W[qb+l] on a uchar pointer), while the inherited Q4_K/Q6_K kernels use block-struct access with ushort reads. Q4_K_M exercises the inherited pair, so the byte-wise kernels are NOT what this measurement indicts — the vectorized ones are already at 16 GB/s. Whether wider loads (uint4/float4 staging) close the remaining gap is unknown and must be bounded by probe before anything is built. Four candidate levers in this campaign have already died that way, and the two that survived were bounded first.

STANDING: this is the first measurement in the campaign that compares GoAI to an incumbent rather than to itself, and it is the one that should set priority. A 7x deficit on the flagship decode path outranks any remaining single-digit-percent lever.

## R-01M01P4XT7EZN9VGWZZ8JQGQ3V Apportioned: non-matmul decode work alone is 2.2x-5.3x llama.cpp's entire token
kind: research
state: draft
created: 2026-08-15

APPORTIONED, and it moves the priority off quant matmul. Measured on TinyLlama-1.1B Q4_K_M, Metal, by toggling the cooperative kernels — which change ONLY the quant matmuls — so the delta bounds their share.

  scalar        9.44 tok/s   105.92 ms/token
  cooperative  24.83 tok/s    40.28 ms/token
  the kernels removed 65.64 ms of 105.92, i.e. 62 percent of a token
  llama.cpp                    5.81 ms/token

THE SPLIT IS SENSITIVE TO ONE UNMEASURED QUANTITY and is reported as a range rather than a point. With total_off = T_matmul + T_other and total_on = T_matmul/k + T_other, the kernel speedup k at THESE shapes is not measured here. k > 2.63 is forced by the observed 2.630x end-to-end ratio — below that T_other goes negative:
     k     matmul now   everything else   vs llama.cpp's WHOLE token
  3.41       27.2 ms         13.0 ms                2.2x
  5.00       16.4 ms         23.9 ms                4.1x
  7.85        9.6 ms         30.7 ms                5.3x

THE ROBUST CONCLUSION, true across that whole range: the NON-MATMUL work alone — attention, norms, RoPE, residuals, sampling, host glue — costs between 13 and 31 ms per token, which is 2.2x to 5.3x llama.cpp's ENTIRE 5.81 ms token. Even with infinitely fast quant matmuls GoAI would land somewhere between 33 and 77 tok/s against llama.cpp's 172.

SO QUANT MATMUL CANNOT CLOSE THE GAP ALONE. That is the opposite of where this campaign has spent its effort, and it is worth stating plainly: the 14 cooperative kernels were a real 3.44x on this model and are not the remaining bottleneck. The previous record read the 7.14x deficit as achieved bandwidth on the weights and implied the kernels were the target; this apportionment shows at most half of it lives there.

WHAT SHOULD BE MEASURED NEXT, in order. First pin k by benchmarking the cooperative Q4_K and Q6_K leaves at TinyLlama's exact shapes (K2048 with N2048 and N5632) without the submit floor, which converts the range above into a point. Then profile the non-matmul path — most of it is attention, RMSNorm, RoPE and residual adds, each of which is a separate dispatch in a command buffer that already batches optimally at one submit per token.

NOT PROPOSED: any specific optimization. Four candidate levers in this campaign died on probe and two survived because they were bounded first.

## R-01M01PHS9AFYJVSYQP10VBN0BW Non-matmul profile: 59 percent of a token is outside the recorded GPU ops; attention is the largest measured term
kind: research
state: draft
created: 2026-08-15

PARTIAL PROFILE, reported with its remainder because the remainder is the finding. Instrumentation reverted; nothing shipped.

MEASURED OP COUNTS per generated token on TinyLlama-1.1B (22 layers), by instrumenting every Recorder entry point during an 8-token generate:
  QMatMulResident 163.8   Binary 82.5   RMSNorm 56.2   Blit 52.1
  MHA 27.5 + MHAAt 27.5   RoPE 25 + RoPEAt 25 + RoPEPair 15   Copy2D 4.5
My earlier hand-estimates were wrong on nearly every line — Binary was assumed 44 and is 82.5, the attention entry points total 55 rather than 22, and RoPE totals 65 rather than 44. Counting beat estimating, again.

MEASURED PER-OP COST, floor-free, at TinyLlama shapes, by the (t16-t1)/15 method that cancels the ~149us per-submit floor:
  QMatMulResident  46,460 ns    MHA 116,651 ns   RMSNorm 17,766 ns
  RoPE             11,006 ns    Binary 7,049 ns  Blit     3,248 ns

BUDGET over an 8-token generation (~322 ms of wall time at the measured 24.83 tok/s):
  QMatMulResident  60.9 ms  19 percent
  MHA              51.3 ms  16 percent
  RMSNorm           8.0 ms   2 percent
  RoPE              5.7 ms   2 percent
  Binary            4.7 ms   1 percent
  Blit              1.4 ms   0 percent
  accounted       131.9 ms  41 percent
  UNACCOUNTED     190.3 ms  59 percent

FIVE-NINTHS OF THE TOKEN IS NOT IN THE RECORDED GPU OPS. That is the result. Candidates, none tested: the incremental method may understate heterogeneous workloads (it measures the marginal cost of a SECOND identical op, which can be cheaper than the first when caches are warm); MHA cost grows with KV length while the probe fixed seq at 64; and there is host-side and synchronisation work between the per-token Commit and Wait that a GPU-op budget cannot see by construction.

WHAT THIS DOES ESTABLISH. Among the ops that ARE measured, attention is the largest non-matmul term at 16 percent and 116,651 ns per call — 2.5x the cost of a quant matmul at the same shapes, on 55 calls per token. If the unaccounted 59 percent turns out to be spread proportionally, attention is the next target; if it is concentrated in host/sync work, none of the GPU kernels are.

NEXT STEP, and it should come before any optimization: a Metal System Trace or GPU counter capture to attribute the 190 ms directly, rather than another arithmetic budget built from micro-benchmarks. Two successive budgets in this campaign have now failed to close — the range-based apportionment and this one — and both failed because they inferred rather than observed.

## R-01M01PPTSTEGFSRG6RTEH4VZHG Observed: 87.6 percent of a decode token is GPU execution — host overhead caps at 12 percent of the gap
kind: research
state: draft
created: 2026-08-15

OBSERVED, not inferred. Two previous budgets in this campaign failed to close because they summed micro-benchmarks; this reads Metal's own per-command-buffer GPUStartTime/GPUEndTime, so it measures what the GPU actually did. Instrumentation reverted; nothing shipped.

MEASURED, TinyLlama-1.1B Q4_K_M, 32 generated tokens:
  wall                        40.47 ms/token   (24.71 tok/s)
  GPU execution               35.47 ms/token   87.6 percent of wall, across 32 command buffers
  host + synchronisation       5.00 ms/token   12.4 percent

THE HOST IS NOT THE PROBLEM. 5.00 ms/token is real and worth having, but eliminating ALL of it caps GoAI at 28.2 tok/s against llama.cpp's 172.1. At most 12 percent of the 7.14x gap is host-side; 88 percent is GPU kernel time.

THAT RESOLVES THE OPEN QUESTION FROM THE PREVIOUS RECORD, and against my own leading candidate. The op budget accounted for only 41 percent of a token and I listed host/sync work as one explanation for the missing 59 percent. It is not: the GPU is busy 87.6 percent of the time. The budget was wrong because the incremental (t16-t1)/15 method UNDERSTATES in-situ cost — it measures the marginal cost of a SECOND identical op against warm caches, while a real token issues heterogeneous ops each touching weights that were not just read. The method is sound for A/B ratios, where the bias cancels, and unsound for absolute budgets, where it does not.

WHAT THE NUMBER MEANS. Over 667 MB of weights re-read per token, GoAI's GPU time corresponds to 18.8 GB/s; llama.cpp's entire token corresponds to 114.8 GB/s. GoAI's GPU work ALONE is 6.1x llama.cpp's whole token.

WHERE THAT LEAVES THE CAMPAIGN. The 14 cooperative kernels tripled this model's decode and are real. The remaining 6.1x is inside GPU kernel execution — split roughly 42/58 between quant matmul and the other GPU ops by the earlier apportionment, both of which are now confirmed to be GPU-side rather than host-side.

METHOD NOTE WORTH KEEPING: when a budget built from micro-benchmarks fails to close, prefer a direct observation of the aggregate over a third refinement of the budget. Metal exposes GPUStartTime/GPUEndTime per command buffer and it took one instrumented function to settle what two rounds of arithmetic could not.

## R-01M01PVFACFE4RDHB9D5GBE062 Attention is 2 percent at decode lengths: 35.81 ms/token is KV-independent, 18.6 GB/s of weight traffic
kind: research
state: draft
created: 2026-08-15

ATTRIBUTED BY OBSERVATION, and it eliminates attention. Instrumentation reverted; nothing shipped.

METHOD. Attention cost grows with KV length while every other per-token op is KV-independent, so sweeping the prompt length and reading Metal's own GPUStartTime/GPUEndTime separates them in situ — no micro-benchmark extrapolation.

MEASURED, TinyLlama-1.1B Q4_K_M, 16 generated tokens per point:
  prompt   4  ->  35.75 ms GPU per generated token
  prompt  64  ->  36.62
  prompt 128  ->  38.39
  prompt 256  ->  40.65
  prompt 448  ->  43.38
Linear fit: GPU ms/token = 35.81 + 0.01747 * KV_len, i.e. 17.47 us per KV position.

ATTENTION IS 2 PERCENT AT DECODE LENGTHS. At the KV~38 of the main benchmark, attention costs 0.66 ms against 35.81 ms of KV-independent work. Even at KV=448 it is 7.83 ms against 35.81, i.e. 18 percent.

THIS CORRECTS THE PREVIOUS RECORD'S LEADING CANDIDATE. That budget, built from the incremental (t16-t1)/15 micro-benchmark, put MHA at 116,651 ns per call and 16 percent of a token, and named attention the next target if the unaccounted time spread proportionally. It does not: the in-situ slope says attention is 2 percent. The micro-benchmark measured MHA at a fixed seq of 64 and, being a marginal-cost method, largely captured launch cost rather than the work a decode step actually issues. Third instance in this campaign of a micro-benchmark budget pointing at the wrong term.

WHAT IS LEFT, stated precisely. 35.81 ms/token of KV-INDEPENDENT GPU work, over 667 MB of weights re-read per token, is 18.6 GB/s. llama.cpp's ENTIRE token is 5.81 ms, i.e. 114.8 GB/s. GoAI's KV-independent GPU work alone is 6.2x llama.cpp's whole token, and it is dominated by moving weights.

WHAT THIS DOES NOT SEPARATE: the 35.81 ms contains the quant matmuls AND the other KV-independent ops (RMSNorm, RoPE, residual adds, the SwiGLU activation), all of which are per-token constants. The sweep cannot split those from each other. What it does establish is that neither attention, nor KV growth, nor host overhead, nor submit batching explains the gap — every one of those has now been measured and ruled out.

NEXT: split the 35.81 ms between quant matmul and the other KV-independent ops by observation. The cooperative on/off toggle changes ONLY the quant matmuls, so measuring GPU time (not wall time) with it on and off attributes their share in situ, without the assumed-k arithmetic that made the earlier apportionment a range.

## R-01M01Q0AG0EHNAC09QEG6YDF6N Diagnosed: quant kernels are per-weight-bound, not bandwidth-bound — achieved GB/s rises with block size
kind: research
state: draft
created: 2026-08-15

DIAGNOSED. GoAI's quant kernels are bound by PER-WEIGHT PROCESSING, not by weight traffic. Observed, not inferred. Instrumentation reverted; nothing shipped.

METHOD. Sweep WEIGHT BYTES at fixed architecture — one 6-layer TinyLlama-shaped model quantized to five types — so op count, layer count and KV length are identical and only traffic changes. GPU time read from Metal's own GPUStartTime/GPUEndTime.

MEASURED, 16 generated tokens per point:
  fmt      MB   ms/token    GB/s   B/256w   bits/weight
  Q2_K  119.4       8.33    14.3       84          2.62
  Q4_K  204.7       6.06    33.8      144          4.50
  Q5_K  250.2       9.71    25.8      176          5.50
  Q6_K  298.5       6.58    45.4      210          6.56
  Q8_0  386.6       7.58    51.0      272          8.50

TIME DOES NOT TRACK BYTES. Q2_K reads the LEAST (119 MB) and takes the LONGEST (8.33 ms); Q8_0 reads 3.2x more and is faster. A bandwidth-bound kernel cannot behave this way, so the intended linear fit (slope = bandwidth, intercept = fixed work) is not applicable — the premise is refuted by the data rather than fitted.

THE PATTERN IS MONOTONIC IN BLOCK SIZE. Achieved bandwidth rises 14.3 -> 51.0 GB/s as bytes-per-256-weights rises 84 -> 272. The denser the packing, the WORSE the throughput. That is the signature of a cost paid per WEIGHT rather than per BYTE: every format decodes the same 256 weights per block, and the formats that spend more instructions doing it are slower regardless of how few bytes they read.

WHY THE EARLIER ALU PROBE MISSED IT. Removing ALL of Q3_K's unpacking arithmetic was worth at most 1.08x, which ruled out the shift-and-mask math. This result is consistent with that and sharper: the per-weight cost is not the arithmetic but the load/issue pattern — one byte-granular access per weight or per few weights, where llama.cpp reads wide vectors and unpacks in registers.

THE CHAIN IS NOW COMPLETE for the 7.14x gap to llama.cpp on TinyLlama-1.1B:
  host and synchronisation          12 percent of wall, ruled out
  submit batching                   already optimal at one command buffer per token
  attention and KV growth           2 percent at decode lengths, ruled out
  weight traffic                    NOT the bound — time does not scale with bytes
  per-weight processing in the      the residual, and the only candidate left standing
    quant matmul kernels

CANDIDATE FOR THE FIRST TIME WORTH NAMING: wide loads. GoAI's kernels index the weight blob as individual uchar/byte reads; the achieved-bandwidth ordering says the cost scales with how many such accesses each weight needs. Staging a block through uint4/float4 reads and unpacking in registers is the standard remedy and is what the ordering predicts would help. It remains UNPROBED, and four candidates in this campaign have died on probe, so it should be bounded before it is built — but unlike those four it is the residual of an elimination rather than a guess.

## R-01M01Q5T70FEDTY330ERTFN7GC Wide loads bounded at 1.34x: the quant kernels are per-weight issue-bound and need vectorizing, not tuning
kind: research
state: draft
created: 2026-08-15

PROBED AND BOUNDED. The wide-loads candidate is worth at most 1.34x, so it cannot close a 6.1x gap. Instrumentation reverted; nothing shipped.

THE PROBE. Complement of the earlier ALU probe: hold grid shape, thread count, loop structure and arithmetic fixed and replace ONLY the weight load with a synthetic value, so every instruction except the memory access remains. Run on Q2_K, the format the previous record identified as worst (14.3 GB/s achieved).

A FIRST ATTEMPT WAS INCONCLUSIVE AND IS RECORDED because the failure mode recurs. Driving the standalone gap benchmark — one op per submit — gave 0.91x with overlapping ranges. At ~258 us per measurement with a ~149 us submit floor, only ~110 us is kernel and the probe's effect was swamped. Re-running through the incremental (t16-t1)/15 harness, which cancels the floor, gave a clean signal. A probe on a kernel must be measured with the floor removed or it measures the floor.

MEASURED, Q2_K cooperative, K2048 N2048, floor removed:
  with real weight loads   41,106 ns/op
  loads replaced by const  30,681 ns/op
  ceiling 1.34x — weight loads are at most 25 percent of kernel time.

COMPONENT CEILINGS NOW BOTH KNOWN:
  remove ALL weight loads     1.34x   (loads <= 25 percent)
  remove ALL unpack ALU       1.08x   (arithmetic <= 7 percent)
  both, optimistically        1.45x
Roughly two thirds of the kernel is NEITHER reading weights NOR unpacking them. What remains per element is the activation load, the fused multiply-add, the loop, and the per-row simd_sum — i.e. instruction issue for a body that processes ONE weight at a time.

WHAT THAT MEANS, and it is the useful conclusion of this whole diagnostic arc: the gap is not a missing optimization in these kernels, it is their shape. Each lane handles one weight per iteration with about two loads and a few ALU ops; llama.cpp's Metal kernels process 4-16 weights per instruction through vector types. No tuning of the current body — wider weight loads included — reaches beyond about 1.45x, so closing 6.1x requires a kernel that is vectorized per weight, not an improved scalar one.

THIS ALSO EXPLAINS THE BLOCK-SIZE ORDERING from the previous record. Achieved bandwidth rose 14.3 -> 51.0 GB/s from Q2_K to Q8_0 because denser formats need more instructions per weight while reading fewer bytes; if the bound were bytes the ordering would be reversed. Instruction issue per weight predicts exactly the observed order.

STATUS OF THE ARC: host (12 percent), submit batching (optimal), attention (2 percent), weight traffic (not the bound), weight-load width (<=1.34x) and unpack arithmetic (<=1.08x) have each been measured and eliminated. The residual is per-weight instruction issue, and the remedy is a structural rewrite rather than a lever.

## R-01M01Q8V37FQ7SYA5HBJRQMPT4 Correction: my probes bounded components WITHIN the loop shape, not vectorization — and kernel style, not density, sets throughput
kind: research
state: draft
created: 2026-08-15

CORRECTION TO MY OWN INFERENCE, plus the evidence that scopes the next build. Nothing shipped.

WHAT I NEARLY CONCLUDED WRONGLY. The previous record bounded weight loads at 1.34x and unpack arithmetic at 1.08x and said roughly two thirds of the kernel is neither. It is easy to read that as "vectorization is bounded at 1.45x too". It is not. Both probes held the LOOP SHAPE FIXED — same 8 iterations per superblock, same index arithmetic, same one-weight-per-iteration body — and varied only what happened inside it. They bound components WITHIN the current shape. Vectorization changes the shape: it cuts iteration count, index computation and instruction issue together, which is precisely the two thirds neither probe touched. The candidate remains UNBOUNDED by anything measured so far.

DIRECT EVIDENCE THAT KERNEL STYLE, NOT FORMAT DENSITY, SETS THE THROUGHPUT. Five formats on one architecture, GPU time from Metal's timestamps:
  fmt    kernel style                 GB/s   bits/weight
  Q2_K   hand-written, byte-wise      14.3          2.62
  Q4_K   inherited, ushort/float4     33.8          4.50
  Q5_K   hand-written, byte-wise      25.8          5.50
  Q6_K   inherited, ushort/float4     45.4          6.56
  Q8_0   inherited                    51.0          8.50
The three kernels I wrote in this campaign index the weight blob byte by byte; the Q4_K/Q6_K kernels inherited from the stranded work use block-struct ushort reads with float4 accumulators, adapted from llama.cpp. Q6_K carries MORE bits per weight than Q5_K (6.56 against 5.50) and is 1.76x faster. Density predicts the opposite; kernel style predicts what is observed.

SO THERE ARE TWO SEPARATE GAPS, and conflating them has muddied the last few records:
  1. My three hand-written kernels (Q2_K, Q3_K, Q5_K) against the inherited style — evidenced at roughly 1.3x to 1.8x, and directly actionable by rewriting them in the inherited idiom.
  2. The inherited kernels themselves against llama.cpp's aggregate 114.8 GB/s — the residual 6.1x on the flagship path, since TinyLlama Q4_K_M exercises only Q4_K and Q6_K, both already in the good style.
Gap 1 does NOT move the llama.cpp comparison at all. It is worth doing on its own merits, for models quantized to Q2_K/Q3_K/Q5_K, and it is cheap because the target idiom already exists in-tree to copy.

NEXT BUILD, scoped: rewrite Q5_K's cooperative kernel in the inherited ushort/float4 idiom, since it is the worst offender with a same-family template beside it (Q4_K differs only by the qh plane). Prediction, from the table: 25.8 GB/s toward the 33.8-45.4 band. That prediction is falsifiable, which is the point — if a style rewrite does not move it, the style hypothesis is wrong and gap 2 needs a different explanation.

## R-01M01QVK03E76BQ2P6DR7FCW42 Vulkan word-hoist is a no-op: its portability uint-array workaround already gave it Metal's 1.72x load pattern
kind: research
state: draft
created: 2026-08-15

NEGATIVE, and the reason is structural rather than a failed idea. Vulkan change reverted; the Metal changes from the previous two commits stand.

THE TRANSFER THAT DID NOT TRANSFER. Widening weight loads gave Metal's hand-written kernels 1.72x (Q2_K) and 1.19x (Q5_K). The same logical change on the Vulkan Q2_K shader — hoisting the shared uint word out of four getByte calls — measured 1.053x with heavily overlapping ranges over six interleaved alternations. Nothing.

WHY, and it is the useful part. The Metal kernels index the weight blob as `device const uchar*`, so each element really was a separate byte load. The VULKAN shaders cannot do that: this backend deliberately avoids the 8/16-bit storage extension for portability, so the weight buffer is declared `uint w[]` and every access goes through
    uint getByte(int off) { return (w[off >> 2] >> ((off & 3) * 8)) & 0xFFu; }
Four consecutive getByte calls therefore index the SAME w[] element, and the compiler already collapses them. The Vulkan shaders have been word-loading all along; there was nothing to hoist.

A PORTABILITY WORKAROUND WAS ACCIDENTALLY THE OPTIMIZATION. The uint-array declaration exists because 8-bit storage is an optional Vulkan feature, not because anyone was tuning loads — and it delivered, for free, the exact access pattern Metal had to be rewritten to get. Worth remembering when the two backends' relative numbers disagree: they are not always running comparable code even when the algorithm matches.

CONSEQUENCE FOR THE CROSS-BACKEND TABLE. Earlier records compared Metal and Vulkan achieved bandwidths per format as if the kernels differed only in dispatch geometry. They also differed in load granularity, which is now known to be worth up to 1.72x on Metal. Any future comparison of the two backends' per-format throughput should account for that.

NOT REVERTED: the Metal Q2_K and Q5_K vectorizations, which are measured, bit-identical and mutation-probed. Only the Vulkan attempt is withdrawn, and it cost one shader edit to learn that the backend was already doing it.

## R-01M01R8RE3ESN9DAP364R8MFJS Q2_K Metal vectorization confirmed end-to-end at 1.22x - and the cross-session table that nearly claimed it wrongly
kind: research
state: draft
created: 2026-08-15

## The invalid comparison, first

After vectorizing the Q2_K and Q5_K cooperative kernels' byte loads to `uint`, I compared end-to-end decode throughput against a table measured in an EARLIER session:

| fmt | before | after | delta | vectorized? |
|---|---:|---:|---:|---|
| Q2_K | 92.69 | 114.31 | 1.233x | yes (1.72x leaf) |
| Q3_K | 59.84 | 67.69 | 1.131x | no |
| Q4_0 | 90.61 | 105.15 | 1.160x | no |
| Q4_K | 108.53 | 123.02 | 1.134x | no |
| Q5_K | 71.81 | 91.16 | 1.269x | yes (1.19x leaf) |
| Q6_K | 73.52 | 82.94 | 1.128x | no |
| Q8_0 | 78.14 | 97.68 | 1.250x | no |

The two vectorized formats top the table, which reads like confirmation. It is not. EVERY format improved, including the five whose kernels were untouched, and Q8_0 (+25%, untouched) moved as much as Q2_K (+23%, vectorized). The machine was in a different state between the two sessions; the change-attributable signal is not separable from that drift. Reporting only the two vectorized rows would have looked like clean confirmation of a real effect.

This is FIRST-BENCHMARK-SAMPLE-IS-NOT-COMPARABLE-001 one level up: there the confound was warm-versus-cold within a run, here it is session-versus-session across runs. The control that catches both is identical - the untouched arms. Five formats served as an accidental control group and reported 1.13-1.25x of pure drift.

## The valid measurement

Reconstructed the byte-wise Q2_K kernel from the vectorized source, then alternated the two variants in ONE session: three full alternations, three samples each after a discarded warmup, 32 generated tokens per sample on a 6-layer 2048-dim Q2_K model.

| variant | run 1 | run 2 | run 3 |
|---|---|---|---|
| byte-wise | 106.05 101.49 102.62 | 101.24 101.68 102.89 | 104.21 102.19 102.89 |
| vectorized | 129.78 124.74 124.97 | 125.20 124.34 124.21 | 126.00 124.65 124.85 |

min(vectorized) = 124.21 > max(byte-wise) = 106.05 - the distributions do not overlap across nine samples each. Ratio of medians: 1.22x.

## Reading it

1.72x at the leaf becomes 1.22x on the decode. That dilution is expected and is itself a check on the number: the quantized matmul is most but not all of a decode step, so an end-to-end gain EQUAL to the leaf gain would have indicated a measurement error, not a better result.

The drift-contaminated table happened to land on 1.233x, within noise of the true 1.22x. That coincidence is worth naming: a broken method can return the right answer. Stopping at the table would have recorded a correct number for a reason that does not hold, and carried the method to the next change, where it would not be so lucky.

## Scope limit

No effect on the llama.cpp head-to-head. TinyLlama Q4_K_M contains no Q2_K tensors, so this gain is invisible there. It is a real win for Q2_K-quantized models and nothing more.

## R-01M01RYSXSE34RF2ZAKM6ZPM8A FMA contraction is a per-site decision, not a source-text property - it broke two bit-identity contracts in nn, plus a dtype assumed by a typed fast path
kind: research
state: draft
created: 2026-08-15

Three failures in nn, all pre-existing on main, all one cause: a perf fast path that stopped matching the reference path it advertises. Two of the three are FMA contraction, which is now a repeat class rather than an incident.

## Why only one was visible

`go test ./nn/` on main reported exactly one failure - the Muon panic - because a panic aborts the whole test binary. MARS and DyT were red the entire time and invisible behind it. A panic in a package suite hides every later failure in that package; the count of red tests on main was understated by 2x until the panic was fixed. Worth generalizing: after fixing any panicking test, re-run the full package before assuming the package is now green.

## 1. Muon: dtype assumed, not checked

`stepParam` called `Storage().F64()` unconditionally. It replaced `AtF64`/`SetF64`, which handle every dtype transparently, so the substitution silently narrowed the supported dtype set from all to one - and `Storage().F64()` PANICS rather than degrading. Muon was simply unusable on F32 parameters.

The general shape: replacing a dtype-agnostic accessor with a typed one is a perf move that carries a correctness precondition the accessor did not have. The precondition must be checked, not assumed, and the fallback must exist.

## 2 and 3. FMA contraction breaks bit-identity contracts

MARS: fast and generic paths disagreed by 1 ulp from step 2. Instrumenting intermediates isolated it - `c` and `v` agreed, `m` did not. `Beta1*m[i] + (1-Beta1)*c[i]` is contracted into a single FMA inside parStep's closure but NOT in the plain generic loop. Identical source text, different fusion decision, therefore different rounding.

DyT: the fused inference path computes `tv*gs[c] + bs[c]`, contracted to one FMA - one rounding, where the OpMul-then-OpAdd path it claims to reproduce does two.

The lesson: **on arm64 the Go compiler may contract `a*b+c` into an FMA, and whether it does is a per-expression-site decision, not a property of the source text.** Two textually identical expressions in different function contexts can round differently. Any code claiming bit-identity across two paths must therefore make rounding explicit, not rely on the expressions matching:

- `math.FMA(a, b, c)` - one rounding, architecturally defined, an intrinsic on arm64 so no cost. Use when you want the fused result.
- `float64(a*b) + c` - two roundings, contraction blocked. Use when reproducing a two-op sequence. The conversion is load-bearing and should be commented as such, or someone will delete it as a no-op cast.

MARS took the first (14 sites), DyT the second (it must match a two-op dispatch chain).

## Note on inlining

Muon's shared momentum fold is a helper inlined at all three dtype sites. Inlining happens before the fusion decision, so inlined copies are separate sites again and could in principle diverge - a shared helper does NOT by itself guarantee identical rounding across call sites. It is safe here only because Muon's per-dtype goldens are compared independently, not against each other.

## Verification

Every fix probed for non-vacuity by reverting it and confirming red. The Muon F32 paths were confirmed reached (panic probe) and load-bearing (mutation).

One probe result is worth recording because it looks like a test weakness and is not: a UNIFORM multiplicative perturbation of the gradient leaves TestMuonStepIsBitIdentical green at ANY magnitude, including 1e-4. `newtonSchulz5` normalizes its input, so Muon is genuinely scale-invariant in the gradient - scaling every element by k leaves the update identical. Only a NON-UNIFORM change (even-index-only 1e-6, or an index shift) turns it red. Both do. Reporting the uniform green as an unguarded path would have been wrong.

## Perf

Interleaved A/B, MARSStep_1M_serial and DyTForward_F64: no measurable regression. new <= old in all three pairs, but both variants drift downward across the run, so the paired deltas sit within that drift. Not a speedup claim.

## R-01M01SQHGFFGTBB2W42QBW82ZB CI has not been running the selected packages' tests - a stray -- swallowed the package list and go test exited 0
kind: research
state: draft
created: 2026-08-15

The repo's selective CI runner has been reporting success without running the tests it selected. Every "run tests for the affected packages" step - pure-go, race, coverage, cgo, vulkan - tested one package and exited 0 regardless of what was broken.

## Mechanism

CI invokes `cichange -run $BASE HEAD -- -short -count=1 -timeout 10m`. Go's `flag.Parse` stops at the first POSITIONAL argument (`base`), so it never consumes the `--` as a flag terminator. `flag.Args()[2:]` therefore hands `["--", "-short", ...]` to Run, which forwards it into the command line: `go test -- -short -count=1 <18 packages>`.

`go test -- ...` does not test those packages. Everything after `--` is passed to the test binary, the package list is SILENTLY DISCARDED, go test falls back to the package in the working directory, and it EXITS 0.

Reduced: `go test -- -short -count=1 ./nn/ ./tensor/ ./classic/` prints `ok github.com/jxsl13/goai` and nothing else.

## Evidence

Commit cd41bf8d (PR #897) merged with 15/15 checks green while leaving `./nn/` panicking on every run. On that exact range:
- `cichange -impact` lists 20 packages including nn - the SELECTOR was correct
- `cichange -run` prints `ok github.com/jxsl13/goai`, exit 0 - the RUNNER discarded the answer

So the impact analysis, which is the sophisticated part and the part that gets attention, was never the problem. A one-token argument-passing bug downstream of it nullified the entire apparatus.

## Consequence

main is red today: `go test -short ./...` on origin/main fails in nn (a panic, which then hides two further failures behind it - a panic aborts the package binary) and in internal/apicheck. CI reported green throughout.

## Fix and guard

Strip a leading `--` in Run. The guard is TestRunPropagatesFailure duplicated with CI's ACTUAL argument vector; the original passes the go-test args bare, which is exactly why it stayed green through the whole outage. Reverting the strip reproduces the outage on demand (exit 0 on a deliberately failing test).

## Transferable lessons

1. A test harness needs a test that exercises the harness AS INVOKED. The existing suite covered Run's behavior thoroughly but always with hand-written arguments, never with the argument vector the CI YAML actually passes. The gap between "the function works" and "the function works the way it is called" is where this lived, undetected, across many PRs.

2. Green CI is evidence about the CI, not about the code, until something independently confirms the tests ran. A selective runner should be assumed broken until it is observed FAILING on a known-bad input. Worth adding as a standing check: periodically feed the runner a deliberately failing package and confirm it goes red.

3. Silent argument swallowing is the dangerous class. `go test` with a bogus package list would have errored loudly; `go test --` succeeds while doing almost nothing. Prefer failure modes that error over ones that no-op.

4. A panic in a package test binary hides every later failure in that package. After fixing any panicking test, re-run the whole package before believing it is green - fixing the nn panic exposed two more pre-existing failures behind it.

Filed as PR #1072; the nn failures it uncovered are PR #1071.

## R-01M01W1ZW4EG8S7YTJ4Y4X0W2M Decode attention had a ~99us sk-independent floor - a runtime loop bound spilled per-thread arrays to memory; constant trip count gives 5.9-8.7x
kind: research
state: draft
created: 2026-08-15

Decode attention on Metal carried an sk-INDEPENDENT floor of ~99us (dk=64) and ~239us (dk=128). A single query row against EIGHT keys cost 99us, which no amount of attention work explains. Fixed: 5.9-8.7x on the kernel, 1.11x end-to-end at short context and growing with context.

## How it was found

Per-kernel dispatch profiling of a 22-layer TinyLlama-shaped Q4_K decode (22.83 ms/token, 43.8 tok/s) gave 306 dispatches/token: QMatMulResident 117.9, Binary 70.1, RMSNorm 47.8, Blit 45.4, RoPEPair 23.4, Copy2D 2.1.

Attention was ABSENT. MHAAt calls C.mtl_recorder_mha directly and does not route through the flashattn wrapper that was instrumented, so the profile silently omitted it. Instrumenting MHAAt added 23.4 dispatches/token - one per layer.

Two candidate explanations were measured and KILLED first:
- per-dispatch overhead: marginal cost of one recorded dispatch is 1.73-2.15us, so 188 non-matmul dispatches cost ~0.4ms, not the missing ~18ms.
- the non-matmul kernels themselves: at decode shapes RMSNorm(rows=1,d=2048) is 4.74us, Binary 1.8-2.35us, Unary 2.54us, Blit 1.41us. All of them together are ~0.5ms/token.

Then MHA at decode shape: 99.36us at sk=8, 119.72 at sk=36, 164.48 at 128, 423.71 at 512, 763.90 at 1024. Slope only ~0.65us/key, INTERCEPT ~94us. A fixed cost that large, per layer, is the defect.

## Cause

mha_decode_f32 declares per-thread `float q[128]` and `float acc[128]` and walks them with `for (int d=0; d<dk; d++)` where dk is a KERNEL ARGUMENT. A runtime trip count means the arrays are dynamically indexed, so they cannot be held in registers and spill to memory. Every acc[d] touch in the key-streaming loop AND in the 5-level simdgroup merge becomes a memory access. The merge walks all dk accumulators at every one of its 5 levels regardless of sk - which is exactly the sk-independent term.

## The hypothesis that looked identical and was wrong

First attempt: assume the problem is array SIZE (1 KB of per-thread arrays) and specialize dk<=64 to q[64]/acc[64]. Interleaved A/B, 3 alternations: 99.52 / 100.10 / 99.43 vs 99.07 / 99.48 / 99.44. Not a 1% difference.

That null result was only trustworthy because the specialization was verified REACHED: mutating the dk<=64 kernel to write zeros turns the attention tests red. Without that probe the correct reading ("size is not the problem") is indistinguishable from "my new kernel is never called" - the same reachability trap as the Vulkan cooperative shaders.

Size was never the issue; DYNAMIC INDEXING was. Compiling the trip count as a constant (d<64) lets the compiler unroll and keep both arrays in registers.

## Result

Interleaved A/B, 3 alternations, non-overlapping distributions:

dk=64:  sk=8   99.8 -> 11.5us (8.7x) | sk=36 120.4 -> 14.4 (8.4x) | sk=128 164.6 -> 22.6 (7.3x) | sk=512 423.6 -> 68.4 (6.2x) | sk=1024 766.5 -> 129.6 (5.9x)
dk=128: sk=36 238.6 -> 64.9us (3.7x) | sk=512 891.0 -> 277.2 (3.2x)

End-to-end, 22-layer TinyLlama-shaped Q4_K decode: 44.3 -> 49.1 tok/s (1.11x), three alternations, non-overlapping. Only 1.11x because the measurement ran at a SHORT context (sk~36) where attention is 2.6ms of 22.6ms. The win scales with sk: at sk=1024 attention per token drops from 22*766us = 16.9ms to 22*130us = 2.9ms.

Routing is gated on dk==64 / dk==128 EXACTLY, not dk<=64. The first cut used <=64 with a hardcoded 64 trip count, which silently computed 64 dims for dk=32 callers and turned TestRecorderDecodeAttn red - caught, then gated exactly.

## Follow-on

The same pattern - a runtime bound over a per-thread array - is worth auditing across the other Metal kernels. It is invisible to inspection (the code looks clean and dimension-agnostic) and costs an order of magnitude.

Also worth noting for future profiling: instrumenting a wrapper is not instrumenting the path. flashattn was counted and read as "attention costs nothing"; the real call went around it.

## R-01M01XARXSEXTRFTS5X60Y65CF The runtime-loop-bound defect is a class: two more Metal attention kernels fixed for 4.6-6.4x, plus a kernel that was live and completely unguarded
kind: research
state: draft
created: 2026-08-15

The runtime-loop-bound defect found in mha_decode_f32 is a CLASS, not an instance. Auditing every Metal kernel that declares a per-thread array found three more; two are fixed here for 4.6-6.4x, one is left.

## The audit

Signature: a per-thread array declared at a fixed maximum, walked with a loop bound taken from a kernel ARGUMENT. Grepping for `float <name>[N]` in metal_bridge.m returns four kernels:

- mha_decode_f32 - fixed previously (5.9-8.7x)
- flashattn_f32 - acc[128], q[128], five `d<dk` loops. This is the kernel behind backend.OpMHA.
- mha_f32 - acc[128], five `d<dk` loops. The Recorder's non-causal sq>1 and windowed path.
- retention_f32 - acc[128] walked with `j<dv`. NOT yet addressed.

Note what the audit also established: mtl_recorder_mha routes causal!=0 to the decode kernel, so PREFILL was already covered by the earlier fix. mha_f32 only serves non-causal sq>1 and window>0.

## Results

dk==64 specializations, 3 alternations each, near-zero variance:

| kernel | shape | before | after | |
|---|---|---|---|---|
| flashattn_f32 | seq=128 | 2612us | 481us | 5.4x |
| flashattn_f32 | seq=384 | 8061us | 1716us | 4.7x |
| mha_f32 | seq=128 | 3937us | 620us | 6.4x |
| mha_f32 | seq=384 | 17803us | 3887us | 4.6x |
| mha_f32 | window seq=512 | 25727us | 5553us | 4.6x |

## Two findings that matter more than the numbers

**1. The public op API cannot see kernel wins.** backend.Execute(OpMHA) uploads Q,K,V and downloads O per call - about 1.2 MB at seq=384. Transfer dominates so completely that the A/B there came out bimodal and UNCORRELATED with the variant: 987 / 636 / 650 / 975 / 987 / 968 us with variants alternating. Measured through that API the 5.4x flash win reads as "no effect".

This is the same regime error as timing a cache-resident weight and concluding a decode kernel is instruction-bound, and it is now the second time it has appeared. The general rule: before trusting a benchmark, ask which resource it is actually exercising. A harness that includes host transfer cannot measure a kernel; a harness whose working set fits in cache cannot measure a streaming kernel.

**2. mha_f32 at dk=64 was live in production and completely unguarded.** Zeroing the entire output of the new dk==64 specialization left `go test -run 'MHA|Attn|Bert'` GREEN. A stderr probe confirmed the specialized pipeline built and took 1620 of 1620 dispatches. So the kernel ran, produced garbage, and no test noticed.

This was only discovered because the mutation probe is now routine after the earlier size-hypothesis episode. Without it the correct reading and the useless reading are indistinguishable. TestMHATwoPassMatchesReference now covers it against a float64 reference over exactly the shapes that reach the kernel, plus dk=32/48 to hold the exact-equality gate. Zeroing the output and dropping one of 64 dims both turn it red. A 1e-4 uniform score scale is ABSORBED and that is correct, not a gap: after softmax a uniform score scale moves outputs ~1e-5, below the 2e-5 f32-vs-f64 tolerance.

## Left open

retention_f32 (`acc[128]` with `j<dv`) has the same defect and is not fixed - RetNet is a lower-traffic path and it deserves its own measurement plus a reachability probe before touching it, on the evidence that one of these three kernels turned out to have no coverage at all.

Also still unexplained from the decode profile: after the attention fix the 22-layer decode is 20.4 ms/token while measured matmul (~4ms) + small kernels (~0.5ms) + attention (~0.3ms) account for about 5ms. Roughly 15 ms/token is still unattributed and is the next thing to chase.

## R-01M01YJPAJF3WADMKEXGXFE89E Decode elementwise ops ran over the whole context buffer for one row - 2.83x (48.7 to 138.0 tok/s); measured components not summing to the whole meant an assumed parameter was wrong
kind: research
state: draft
created: 2026-08-15

The decode path spent ~65% of its GPU time on elementwise ops running over the WHOLE context-sized buffer for a single decoded row. Bounding them to the rows actually in play: 48.7 -> 138.0 tok/s, 2.83x, bit-identical output.

## The defect

Decoder buffers are allocated for ctx rows so one Decoder serves both prefill and decode. Every recorded op takes a row count - RMSNorm, RoPE, MHA, AddBias - EXCEPT Binary, which derived its length from the buffer.

For one decoded row each residual add therefore touched ctx*dim = 1024*2048 = 2,097,152 elements, and each SwiGLU ctx*hidden = 5,767,168. A 22-layer decode issues ~46.8 adds and ~23.4 SwiGLUs per token: ~233M elements, roughly 2.8 GB of traffic per token, against ~520 MB of actual weight reads. The elementwise ops moved 5x the memory traffic of the entire quantized model.

## How it was found - the reusable part

Three measurements, each eliminating a hypothesis:

1. GPU-vs-wall split via command-buffer GPUStartTime/GPUEndTime: 17.48 of 20.26 ms/token was GPU-busy, 1 command buffer per token. NOT host overhead.
2. Realistic matmul workload - 154 projections over 520 MB of DISTINCT resident weights (defeating the cache reuse the roofline test enjoys): 4.10 ms at 132.8 GB/s. So the matmuls were as estimated.
3. Serializing those matmuls through a shared output buffer: 4.12 ms vs 4.18 ms independent. NOT the dependency chain.

That left ~13 ms of GPU work in kernels I had measured at ~0.5 ms in isolation. The contradiction resolved only by logging the ACTUAL element counts the recorder was called with:

  Binary calls/token=70.1  n=2097152 x46.8  n=5767168 x23.4

Every previous estimate multiplied a measured per-op cost by a call count, using the shape the op was ASSUMED to run at. That estimate was wrong by three orders of magnitude, and no amount of refining the per-op microbenchmarks would ever have found it - the microbenchmark used the assumed shape too.

**The lesson: when measured components do not sum to the measured whole, the error is more likely in an assumed PARAMETER than in the per-component timings. Instrument the actual arguments before re-measuring the pieces.** Three rounds of this investigation refined per-op costs that were individually accurate and collectively meaningless.

## Fix

Recorder.BinaryN bounds the op to the first n elements. The C side already accepted n; only the Go wrapper hardcoded o.n. Reached through an OPTIONAL binaryNRecorder capability, the same shape as the existing quantAccRecorder, so recorders without it keep the unbounded call and stay correct. linear.recordAdd gains the residual width; all seven call sites write into dx, whose row width is d.d.

## Result

Interleaved A/B, 3 alternations, non-overlapping:

  unbounded    48.7 / 48.8 / 48.6 tok/s
  all-bounded 138.3 / 137.8 / 137.8 tok/s   = 2.83x

Residual adds alone: 62.6 tok/s. SwiGLU is the larger half of the traffic (135M of the 233M elements).

Correctness: generated token stream BIT-IDENTICAL over 64 tokens from an 8-token prompt, exercising prefill (rows>1) and decode (rows=1). Bounding only skips buffer-tail elements no consumer reads.

## Position

TinyLlama-shape decode now ~138 tok/s against llama.cpp's 172 on the real model - the gap narrows from 7.14x to about 1.25x. Combined with the session's other decode work, and with the Q4_K matmul already at 93% of the M2 Pro memory roofline, the remaining distance is small enough that the next step should be a fresh GPU-vs-wall split and element-count log on the NEW profile rather than any assumption carried over from this one.

## Not done

Unary has the identical unbounded shape and is not fixed - it does not appear in a Llama decode (the profile shows zero Unary calls), so it needs a workload that reaches it (GELU/ReLU2 FFNs, Mamba, RWKV) before it can be measured. Same for the remaining Binary sites in the Mamba/RWKV paths. retention_f32 still carries the runtime-loop-bound defect from the earlier audit.

## R-01M01Z6AC8F42BT1ZPEH18K497 Decode is now host-bound (GPU 69% of wall); a real 5x host-encode win changed nothing because encode overlaps GPU - optimize only what is on the serial critical path
kind: research
state: draft
created: 2026-08-15

A fresh profile taken AFTER the elementwise fix, deliberately carrying no assumptions from the previous one. The decode is now host-bound, not GPU-bound, and the obvious host optimization turned out not to matter.

## New profile

  wall 7.17 ms/token | GPU-busy 4.97 ms/token | GPU 69.3% of wall | 1 command buffer/token | 139.5 tok/s

GPU share fell from 86.3% to 69.3%. Element counts per token are all small again: QMatMulResident 0.49 Melem, BinaryN 0.25, RMSNorm 0.11, RoPEPair 0.07, MHAAt 0.05, Blit 0.01, Copy2D 0.01.

Measured matmuls are 4.10 ms of the 4.97 ms GPU time, so the GPU side is now ~82% matmul and close to structurally optimal. The remaining GPU headroom is the gap between the realistic matmul sequence (132.8 GB/s over 520 MB of distinct weights) and the single-weight roofline (186 GB/s) - about 1.2 ms if fully closed.

The larger target is the ~2.2 ms/token of wall-minus-GPU.

## What was tried

Per-dispatch HOST encode cost measured at 2.89 us. Cause: every recorded op allocated a fresh MTLBuffer via newBufferWithBytes purely to carry a few ints. That is a driver allocation on the encode path, ~330x per token. Metal's setBytes inlines constants under 4 KB into the command buffer and allocates nothing.

Converted 25 sites, removed 23 allocations. Host encode cost 2.79-3.86 -> 0.59-0.72 us/dispatch, about 5x, three alternations.

## And why it did not help

End-to-end: 138.2 / 142.1 / 139.8 vs 139.4 / 143.1 / 138.5 tok/s. Within noise.

Encoding OVERLAPS GPU execution (the T614 encode-overlap split: the host encodes the next buffer while the current one runs). At GPU 4.97 ms/token against ~0.95 ms of encode, the entire encode phase is hidden. Removing 80% of a hidden cost changes nothing.

This is a third instance of the same family of error, and the most instructive one because the leaf measurement was completely valid: 5x is real, reproducible, and irrelevant. The earlier two were measuring the wrong RESOURCE (cache vs DRAM) and the wrong PATH (host-transfer-bound op API). This one measures the right thing on the critical path's ingredients but not on the critical PATH - the work is off it.

**Rule: before optimizing a cost, establish that it is on the serial critical path. A leaf speedup on overlapped work is unfalsifiable at the leaf and worthless at the system.** The cheap check is the one already in hand: wall minus GPU-busy tells you how much serial host time exists at all, and encode is only part of it.

Kept anyway - strictly less work, 23 fewer driver allocations per dispatch path, and it stops being hidden as GPU time shrinks. Explicitly NOT presented as a throughput win.

## Next

The ~2.2 ms is SERIAL host time and, by elimination, is not encode. Candidates in order: logits download (32000 floats/token), the greedy argmax, command-buffer submission and completion latency (1 buffer/token, so ~2.2 ms of scheduling would be enormous - worth measuring before assuming), and Go-side per-step bookkeeping. Instrument the Generate loop phases directly rather than inferring, which is the lesson already paid for once this session.

## R-01M01ZCXKJFZ68MFV7HS1STABC Decode is now matmul-bound: Step is 94% of wall, sampling 1.4%; the remaining structural gap is small-projection matmul bandwidth (132.8 vs 186 GB/s)
kind: research
state: draft
created: 2026-08-15

Instrumented the Generate loop phases directly, as the previous record said to do rather than infer. The ~2.2 ms of wall-minus-GPU is NOT sampling and NOT the host loop. It is inside the step itself.

## Measurement

22-layer TinyLlama-shape Q4_K, 64 tokens, timers inside Generate:

  d.Step (encode + submit + GPU + logits download)   5.95 ms/token   94%
  host sampling incl. f32->f64 convert over 32000     0.09 ms/token    1.4%
  Generate total                                      6.32 ms/token   (158 tok/s)

Against GPU-busy of 4.97 ms/token, so roughly 1 ms sits inside Step but outside GPU execution: submission and completion latency plus the 128 KB logits download.

## Two hypotheses killed

1. **Sampling.** A greedy argmax over 32000 logits plus the float32->float64 conversion costs 0.09 ms/token, 1.4% of wall. It was a plausible suspect (32000 elements per token, strictly serial, on the critical path) and it is irrelevant.

2. **Device top-k.** I predicted Generate was paying for dt.TopKN on the fast sampling path, because a probe loop calling d.Step directly measured 5.43 ms against Generate's 6.32 and TopKN was the obvious difference. Wrong: the probe printed fastPath=false. nlp.Greedy() does not satisfy fastTopKSampler, so Generate takes the same host-download path as the probe. The gap was run-to-run variance, not a missing kernel. Worth recording because the reasoning was sound and the conclusion was still wrong — the instrument, not the argument, settled it.

## Where the time is now

  wall ~6.3-7.2 ms/token (run to run) | GPU-busy 4.97 | matmuls 4.10 of that

So: matmuls are ~82% of GPU time and ~65% of wall. Everything else — all elementwise, all norms, all attention, all sampling, all host work — is the remaining third, and no single item in it exceeds ~1 ms.

The decode is now essentially matmul-bound, which is the correct end state for a weight-streaming decode.

## The one remaining structural gap

The realistic matmul sequence runs at 132.8 GB/s (154 projections, 520 MB of distinct weights) against a single-weight roofline of 186 GB/s. Closing that fully would take 4.10 ms to ~2.9 ms, i.e. roughly 6.3 -> 5.1 ms/token, about 195 tok/s.

The deficit is concentrated in the SMALL projections: the roofline sweep shows 108 GB/s at a 2.2 MB weight rising to 191 GB/s at 18 MB. Per layer, q and o are 2.2 MB each while gate/up/down are 6.2 MB each, so the two small ones are disproportionately inefficient. This is a launch/ramp effect at small dispatch sizes, not an algorithmic one, and the plausible lever is batching the small projections into fewer dispatches (q and o are independent of each other only across sublayers, but k and v are genuinely fusable with q as a single QKV projection — worth checking whether that fusion already exists on this path).

## Position

~154-158 tok/s on a TinyLlama-shaped Q4_K decode against llama.cpp's 172 on the real model. Same shape, random weights, same machine — but NOT the same binary or the same measurement harness, so this is an indicative comparison and not a head-to-head. A real head-to-head on the actual TinyLlama GGUF is the honest next validation, and it should be run before any claim about relative standing.

## R-01M01ZEHCKEDVSNM13HEKZR9WD Correction: QKV fusion already exists, so it is not an available lever; and the 4.10 ms matmul figure was built from assumed rather than logged dispatch shapes
kind: research
state: draft
created: 2026-08-15

Correction to R-01M01ZCXKJFZ6's closing suggestion, and a caveat on the 4.10 ms matmul figure.

## QKV fusion already exists

recordQKVProj records ONE projection through b.wqkv into a fused [·, qDim+2*kvDim] buffer; RoPEPair and MHAAt then address bands of it by offset (the §T613 fused-QKV view). So "fuse q/k/v to reduce small dispatches" is not available as a lever — it is already done. The profile's ~5.4 QMatMulResident per layer is consistent with qkv + o + gate/up + down rather than seven separate projections.

## Caveat on the 4.10 ms matmul measurement

That number came from TestZZLayerSeq, which built 22 layers x SEVEN separate projections at the shapes I assumed a Llama layer has. The real decoder issues fewer, larger dispatches. Total weight BYTES are identical either way — the same weights, packed differently — so ~4.1 ms remains approximately right as a bytes/bandwidth estimate. But the per-dispatch efficiency is not the same, and since small dispatches are exactly where the bandwidth deficit lives (108 GB/s at 2.2 MB vs 191 GB/s at 18 MB), the real mix is likely somewhat BETTER than the synthetic one.

This is the same error as the elementwise episode: a benchmark built from assumed shapes rather than logged ones. It did not mislead the conclusion this time (the number is close, and it was cross-checked against GPU-busy time), but the estimate should be rebuilt from the logged dispatch shapes before anyone optimizes against it.

## Where the headroom actually is

Total decode weight traffic is ~580 MB/token (543 MB of layers + ~37 MB output head). GPU-busy is 4.97 ms/token, i.e. ~117 GB/s effective including all non-matmul GPU work. At the measured 186 GB/s streaming roofline, 580 MB would be 3.1 ms; adding the measured non-matmul GPU work (~0.5 ms) gives ~3.6 ms against the current 4.97 ms.

So there is roughly 1.4 ms/token of GPU headroom, worth about 6.3 -> 4.9 ms/token, or ~205 tok/s. It is a bandwidth-efficiency problem at realistic dispatch sizes, not an algorithmic one, and it is the last structural item in the decode.

## Honest standing

~154-158 tok/s on a TinyLlama-SHAPED Q4_K decode with random weights, against llama.cpp's 172 on the real TinyLlama Q4_K_M. Same machine and same shape, but a different binary, different weights and a different harness. That is indicative, NOT a head-to-head, and no claim of parity or superiority should be made from it. The honest validation is the existing llamagpu/tinyllama_vs_llamacpp_test.go harness on the real GGUF, re-run end to end; that is the next thing to do before any further optimization.

## R-01M01ZZMYFFEKVTJJPQ3EGVM2J Real head-to-head: GoAI 143.17 vs llama.cpp 201.61 tok/s = 0.71x — the hardcoded 172.19 baseline had rotted and the synthetic shaped model was 8-10% optimistic
kind: research
state: draft
created: 2026-08-15

The real head-to-head, run as the previous record said it must be before further optimization. It corrects the standing claim in two directions at once.

## Numbers

llama-bench and GoAI on the SAME TinyLlama-1.1B Q4_K_M file (636 MiB, 1.10 B params), M2 Pro, Metal:

  cold, separate runs:  GoAI 143.17  llama.cpp 201.61 +/- 3.25  = 0.710x
  hot, single run:      GoAI 112.84  llama.cpp 155.09           = 0.728x

Absolute values differ by 20% between the two runs from thermal drift; the ratios agree to 2.5%. The ratio is the quantity that survives, which is why both sides must be measured in the same session.

## Correction 1: the incumbent moved

The harness hardcoded llama.cpp at 172.19 tok/s, recorded for build 48d22e295. Today, same host, same file, same command: 201.61. Every ratio computed against that constant flattered GoAI by ~17%.

This is the exact failure the classical-ML scorecard hit before (T881/B103: an unrecorded sklearn version turned a "beats every method" claim into an honest split). The lesson had been written down and the harness still rotted, because recording a version stamp does not stop a constant from aging — only re-measuring does. The fix is structural: llamaCppTG64 now RUNS llama-bench when it is on PATH and parses the tg64 row, using the recorded constant only as a named fallback so a reader can always tell a live comparison from a stale one.

## Correction 2: the synthetic model was optimistic

All the optimization work this session was measured on a TinyLlama-SHAPED model with random weights: 154-158 tok/s. The real model measures 143.17. Same architecture, same quantization, ~8-10% apart — close enough to have guided the work correctly, far enough that quoting it as "the TinyLlama number" would have been wrong.

## Standing, honestly

GoAI is 1.41x BEHIND llama.cpp on this workload, not the ~1.1x that the stale constant plus the synthetic model together implied.

Session progress on the same harness and the same real model is nonetheless real: 24.11 -> 143.17 tok/s, 5.94x. Against the then-current llama.cpp the gap was 7.14x; against today's it is 1.41x.

## What the remaining 1.41x is

From the phase split: decode is matmul-bound, matmuls ~82% of GPU time and ~65% of wall, and the realistic matmul sequence runs at ~117-133 GB/s against a measured 186 GB/s single-weight streaming roofline. Closing that bandwidth gap is worth roughly the whole remaining factor. It is a dispatch-efficiency problem at realistic weight sizes (108 GB/s at a 2.2 MB weight vs 191 GB/s at 18 MB), not an algorithmic one.

QKV fusion is already implemented, so the obvious "fewer, bigger dispatches" lever is spent. The open question is why a 2-6 MB resident weight streams at ~60-70% of the rate a 144 MB one does, and whether that is fixable from the kernel side (occupancy, ramp) or is a fixed per-dispatch cost that only batching across layers could amortize.

## R-01M020BCTDFRS9FZD22V2GTSBV Small-dispatch matmul deficit isolated (100.5 vs 146.7 GB/s) — but halving rowsPerSimd does not fix it, and the measurement noise floor exceeds the effect
kind: research
state: draft
created: 2026-08-15

Isolated the small-dispatch bandwidth deficit properly, then tried the obvious fix and it failed. Recording both.

## The deficit, measured without the cache confound

Earlier roofline sweeps varied N on a SINGLE weight, so small N was cache-resident and unreadable. This sweep holds total bytes constant (~420 MB) and allocates enough DISTINCT weights at each N that none can be cached:

| N | weights | threadgroups/dispatch | GB/s |
|---|---|---|---|
| 2048 | 186 x 2.2MB | 512 | 100.5 |
| 4096 | 93 x 4.5MB | 1024 | 120.9 |
| 8192 | 46 x 9.0MB | 2048 | 146.7 |
| 16384 | 23 x 18MB | 4096 | 136.3 |
| 65536 | 5 x 72MB | 16384 | 128.8 |

Small dispatches lose about 1.46x against the N=8192 peak. The decoder's o and down projections are both N=2048, i.e. in the worst regime; qkv is 2560 and gate/up 11264.

Note the curve is NOT monotonic in parallelism — it peaks at N=8192 and DECLINES at 16384 and 65536. That already argues against "more threadgroups is better" as the explanation, and it is the reason the obvious fix was worth testing rather than assuming.

## The fix that did not work

Hypothesis: at N=2048 the dispatch is too small to fill the GPU, so halving rowsPerSimd (2 -> 1) doubles the threadgroup count and should recover the gap.

Interleaved A/B, 2 alternations:

  N=2048   rows=2: 107.2 / 117.2   rows=1:  96.8 / 122.5
  N=8192   rows=2: 144.2 / 150.5   rows=1: 135.5 / 141.9
  N=65536  rows=2: 128.0 / 124.4   rows=1: 148.0 / 152.1

At the target case the two configurations OVERLAP — no reliable win. rows=1 helps only at N>=65536, where the decoder never operates, and hurts at N=8192. Reverted.

Run-to-run spread at a fixed configuration is ~10% (107.2 vs 117.2), which is larger than any effect being sought; a conclusive version of this sweep needs far more repetitions than the 8-iteration minimum used here. That is itself a finding: this measurement is not yet precise enough to drive a kernel decision, and the next attempt should establish the noise floor before interpreting differences.

## A maintenance hazard found on the way

rowsPerSimd is duplicated by the dispatch grid in THREE places: coopRows at the two resident sites, and a HARDCODED (N+3)/4 at the standalone site. Changing the kernel constant with only the first two updated under-dispatches and leaves tail rows unwritten — observed as garbage in rows 4..7 of an N=8 case.

The Q4_K parity tests cover N=8 and caught it immediately, so it is a hazard rather than a live defect. Documented in the kernel; a shared constant would be better but is a wider refactor across all seven quant formats.

## Where this leaves the gap

GoAI is 1.41x behind llama.cpp (143.17 vs 201.61 tok/s on the real TinyLlama Q4_K_M). Decode is matmul-bound and the matmuls run at roughly 100-147 GB/s depending on dispatch size, against a 186 GB/s streaming roofline.

The remaining lever is genuinely split-K: partition the K dimension across threadgroups so a small-N matmul gets more parallelism WITHOUT shrinking the per-threadgroup row count, then reduce the partials. That is a real kernel redesign (needs either atomics or a second pass) and should not be started until the measurement noise floor is under control, because its expected effect is the same order as the current run-to-run spread.

## R-01M020R0S1FB2TC5KS28WEH7Y1 Precise GPU-timestamp instrument built (cv 1-2%): split-K repriced from ~1.46x to 9% end-to-end, and the greedy host round-trip killed as a candidate
kind: research
state: draft
created: 2026-08-15

Built the precise instrument the previous record demanded, then used it to REPRICE the remaining work. Two candidate optimizations were killed on measurement, and the biggest one turned out to be worth far less than assumed.

## The instrument

Wall-clock timing gave 100.5 / 107.2 / 117.2 GB/s for the SAME workload — 17% spread, and a non-monotonic dispatch-size curve that implied an effect where there was none. Timing the command buffer by its own GPU timestamps (GPUEndTime-GPUStartTime) removes host submit and wake jitter. With a warmup and 40 samples: cv 1.1-3.0% within a run, warmed runs agreeing to ~0.2%.

The corrected curve is monotonic, and decode-sized dispatches are less bad than wall clock claimed (124.0 vs the 100.5 previously reported):

  N=2048 124.0 | 2560 132.5 | 4096 144.7 | 8192 156.1 | 11264 162.7 | 16384 169.7 GB/s

## Repricing split-K

Against the decoder's REAL shapes — qkv N=2560, o N=2048, gate|up N=11264, down N=2048/K=5632 — a PERFECT small-dispatch fix takes matmul from 3.81 to 3.21 ms/token: 6.99 -> 6.39 ms, 143 -> 157 tok/s, gap 1.41x -> 1.29x.

Nine percent, for a kernel redesign needing atomics or a second reduction pass. Still positive, but no longer the obvious priority, and now a measured trade rather than a guess. Worth noting the earlier wall-clock number would have promised ~1.46x on the worst shape and badly oversold it.

## The arithmetic that reframes everything

GoAI 6.99 ms/token, of which matmul is ~3.81 ms — so ~3.18 ms is overhead. llama.cpp's ENTIRE decode is 4.96 ms/token. GoAI's matmuls are not the problem; its overhead is nearly as large as llama.cpp's whole step.

So the target is the 3.18 ms, not the 0.60 ms available in matmul bandwidth.

## Two overhead candidates killed

1. **Host logits round-trip.** Greedy sampling was falling to the host path — fastTopKSampler rejects TopK<=0 and Greedy sets TopK 0 — so every token downloaded the full [32000] logits row to run an argmax the device could do. Greedy IS top-1, and routing it through the device top-k is exact (sampleTopKCandidates walks candidates in ascending vocab-index order and takes the first maximum, exactly as the full-vocab host argmax does). Verified: BIT-IDENTICAL token stream on the real TinyLlama GGUF.

   Result: 138.25 / 137.85 / 137.75 (host) vs 138.10 / 137.49 / 137.30 (device), interleaved. No win — marginally slower. The device top-k kernel costs about what the 128 KB download it replaces costs. Reverted; shipping it would have been an unjustified change on a speculative benefit.

2. **Host encode** was killed last round for the same class of reason (real 5x leaf win, fully overlapped with GPU execution).

## What is left unexplained

Sampling 0.09 ms, encode overlapped, logits round-trip neutral. That still leaves roughly 2 ms/token of non-GPU time unaccounted. The remaining hypothesis is command-buffer submit-to-start and completion latency: one buffer per token with a strict wait means the GPU idles between tokens, and nothing measured so far covers that interval. Measuring it needs the gap between one buffer's GPUEndTime and the next's GPUStartTime — now directly available through the timestamps this record adds, and that is the next measurement.

If that interval is ~2 ms it is the single largest item in the decode and dwarfs split-K; if it is small, then the 4.97 ms GPU-busy figure and the 3.81 ms matmul estimate disagree and the matmul estimate must be rebuilt from logged dispatch shapes rather than the bandwidth model.

## R-01M02154F2EY5TTQBXKFQSQFN3 Decode is at PARITY with llama.cpp (0.885x, inside its own 19% spread) — the 1.41x was prefill charged to GoAI but not the incumbent; the real gap is prompt processing at 17.4x
kind: research
state: draft
created: 2026-08-15

The decode is at parity with llama.cpp. The 1.41x deficit reported last round was a measurement error: GoAI was being timed with prefill included, against a metric that excludes it. The real deficiency is prompt processing, 17.4x behind, and tg64 never showed it.

## Resolving the fork

The previous record posed a fork: either the ~2 ms/token of non-GPU time is command-buffer submit latency, or the matmul estimate is wrong. Measured both:

  GPU idle BETWEEN command buffers: 0.19-0.59 ms/token (not 2 ms — hypothesis rejected)
  synchronous device->host readback: 2.1 us for 125 KB (unified memory; not a factor)

Neither. The accounting then closed exactly on a bare Step loop — wall 5.811, busy 5.352, idle 0.406, OTHER 0.053 ms/token — while Generate over the same 64 tokens showed other = 1.449 ms/token. Same GPU busy, same idle. The residual was entirely OUTSIDE the timed loop, i.e. the prefill call before it: ~93 ms for a 6-token prompt, amortized across 64 generated tokens.

## The comparison was not like-for-like

llama-bench -p 0 -n 64 reports tg64, which is token GENERATION only. Timing dec.Generate charges GoAI for prompt processing that the incumbent is not charged for.

Corrected, both phases measured live against the same file:

  tg64 (decode)  GoAI 170.78  llama.cpp 192.88   = 0.885x
  pp64 (prompt)  GoAI 102.0   llama.cpp 1778.75  = 0.057x

## Decode is at parity

llama.cpp's own tg64 on this host, six runs: 170.02, 172.40, 173.08, 177.80, 187.35, 201.61 — a 19% spread, per-run stddev 5-9%. GoAI decode-only: 177.9, 178.2, 179.5 — 0.9% spread.

GoAI sits inside llama.cpp's distribution and is markedly more consistent. The 201.61 figure that anchored the last two records was the FIRST and coldest run; using a single cold sample of the incumbent as the baseline overstated the gap. This is the third form of the same error this session — after cache-vs-DRAM regime and hidden-vs-serial cost — and the general shape is now clear: **check that the two things being compared are the same quantity, measured the same way, under the same conditions.**

## The real gap

Prompt processing: 9.8 ms per prompt token against llama.cpp's 0.56, i.e. 17.4x. Prefill runs with M>1, which does not reach the cooperative M=1 kernels the decode work this session was spent on. Worse, prefill has a property decode does not: with a batch of tokens the weights are read ONCE for the whole batch, so it should be compute-bound rather than bandwidth-bound. GoAI is evidently not exploiting that — a 17x deficit is the signature of re-streaming weights per token rather than blocking over the batch.

That is now the largest measured gap in the library, and it is a far better target than split-K (repriced last round at 9% of decode). It also matters more in practice: long prompts are the common case for RAG, code and summarization, and a 17x prefill penalty dominates time-to-first-token.

## Corrected standing

  decode  : parity with llama.cpp (0.885x, inside its 19% run-to-run spread)
  prefill : 17.4x behind
  session : 24.11 -> ~178 tok/s decode on the real model, 7.4x

Next: characterize the M>1 quantized matmul path. Establish whether it reads weights once per batch or once per token, then decide between a blocked/tiled M>1 kernel and routing prefill through the existing MPS/GEMM path.

## R-01M021SRV6FQZR7E28DHRNSATW Prompt processing 3.4x: the cooperative quant kernels were gated on M==1 although they always indexed rows from group.y — and the M>1 fallback was slower per row than not batching
kind: research
state: draft
created: 2026-08-15

Prompt processing was 17.4x behind llama.cpp. A one-condition change took it to 5.2x. The cooperative quant kernels were gated on M == 1 for no reason the kernels required.

## Diagnosis

Time vs batch size on Q4_K (K=2048, N=5632), GPU timestamps:

| M | us/op | us/row | vs M=1 | weight GB/s |
|---|---|---|---|---|
| 1 | 35.2 | 35.2 | 1.00x | 184.5 |
| 2 | 267.9 | 133.9 | 7.62x | 24.2 |
| 8 | 764.8 | 95.6 | 21.75x | 8.5 |
| 64 | 4832.1 | 75.5 | 137.43x | 1.3 |

M=1 runs at 184.5 GB/s, essentially the streaming roofline. M=2 costs 7.6x M=1 for 2x the work, and the per-ROW cost gets WORSE, not better, as the batch grows. A batched path that fails to amortize would be flat per row; this one is worse per row than not batching at all.

Confirmed directly — batched dispatch versus looping the M=1 kernel over the same rows:

  M=2  267.9us vs 69.9us   batched 3.83x SLOWER
  M=8  764.2us vs 280.7us  batched 2.72x SLOWER
  M=64 4829us  vs 2261us   batched 2.14x SLOWER

The batched path lost to simply calling the decode kernel M times.

## The fix

The cooperative kernels take their row index from group.y (mi) and the dispatch already passes M as the grid's y extent, so they were ALWAYS capable of M>1. The gate `cooperative = M == 1` was artificial. Relaxed to M <= 512, verified for all seven formats by checking each kernel indexes mi:

  M=4  508.2 -> 124.6us  4.08x
  M=16 1295.9 -> 490.4us 2.64x
  M=64 4829.4 -> 1953.9us 2.47x
  M=512 37585 -> 15601us  2.41x

End to end on the real model: pp64 102 -> 343.5 tok/s (3.4x); tg64 unchanged at ~174.

Correctness: generated tokens IDENTICAL over 64 tokens from a 32-token prompt, both for Q4_K alone and with all formats relaxed. Prefill logits differ by 7.4e-6 relative (f32 reassociation, different accumulation order), inside the 2e-5 parity tolerance, and it moves no argmax.

## What this is NOT

This does not make prefill batched. The cooperative kernel re-streams the weight for every row — it is the DECODE shape applied M times. That is why the gain plateaus at ~2.4x rather than growing with M: at M=64 the kernel still moves 64x the weight bytes a batched kernel would.

llama.cpp does pp64 at 1778 tok/s against GoAI's 343.5, so 5.2x remains. Closing it needs a genuinely blocked M>1 kernel: tile the batch so each weight super-block is read once into threadgroup memory and reused across all rows in the tile. That is the real prompt-processing kernel and it does not exist yet.

## Lesson

The gate had been there long enough to look intentional, and the surrounding code (coopRows, the M==1 comments) reinforced it. What exposed it was not reading the code but measuring a shape nobody benchmarked: every prior measurement this session was M=1, because decode is M=1. The 17.4x prefill deficit was invisible until tg64 and pp64 were separated — and it had been sitting behind a metric that structurally could not show it.

## R-01M021YMW9FE58W4X81GSPZPD6 The batched quant matmul is COMPUTE-bound at 11% of FLOP peak, not bandwidth-bound — the proposed weight-tiling kernel would have optimized the wrong resource; the fix is simdgroup_matrix
kind: research
state: draft
created: 2026-08-15

Measured the batched quant matmul's limiting resource BEFORE building the tiled kernel the previous record proposed — and the proposal was wrong. The path is compute-bound, not bandwidth-bound, so blocking the weight reads would have optimized a resource that is not the constraint.

## The measurement

Q4_K, K=2048, N=5632, GPU timestamps, after the M>1 gate fix:

| M | us | implied weight GB/s (if re-streamed per row) | GFLOP/s | % of 6.8 TFLOP/s |
|---|---|---|---|---|
| 1 | 35.2 | 184.3 | 655 | 9.6% |
| 4 | 124.6 | 208.3 | 741 | 10.9% |
| 16 | 490.4 | 211.7 | 753 | 11.1% |
| 64 | 1953.9 | 212.5 | 756 | 11.1% |
| 512 | 15600.9 | 212.9 | 757 | 11.1% |

Two things follow.

**The weight is not actually re-streamed M times.** The implied per-row weight bandwidth reaches 212 GB/s, which EXCEEDS the M2 Pro's ~200 GB/s memory ceiling. That is only possible if the repeated reads are being served from cache. The weight for one projection is 6.49 MB and fits comfortably; each threadgroup walks it once per query row and the cache absorbs the repeats.

**Throughput is flat at 11.1% of FLOP peak** from M=4 to M=512. A bandwidth-bound kernel would show falling GFLOP/s as M grows and the working set spills; a compute-bound one is flat. It is flat to three digits across a 128x range of M.

## Why the planned kernel was the wrong build

The previous record proposed tiling the batch so each weight super-block is read once into threadgroup memory and reused across rows. That optimizes weight traffic — which the numbers above show is already cache-served and not the limit. Building it would have cost a substantial kernel rewrite, carried real correctness risk, and delivered close to nothing.

Also worth noting: my first instinct on inspecting the kernel was that the y (activation) traffic would explode under tiling — total y reads scale as N*M*K*rowsPerSimd/coopRows, which at N=5632, M=64 already exceeds the weight traffic. That concern was real but secondary; the FLOP measurement settles the question outright.

## What actually limits it

At 757 GFLOP/s against ~6.8 TFLOP/s the kernel uses about a ninth of the machine's arithmetic. The work per output element is dominated by dequantization: unpacking 4-bit nibbles with shifts and masks, then scalar FMAs, one lane at a time. Every multiply is a scalar f32 op issued from the general ALU.

llama.cpp reaches pp64 = 1778 tok/s against GoAI's 343.5 — 5.2x — and the mechanism is Apple's matrix units. metal_bridge.m contains ZERO uses of simdgroup_matrix or simdgroup_multiply_accumulate; every kernel in the file is scalar/SIMD-lane arithmetic. For M=1 that is correct (decode is bandwidth-bound at 92% of the memory roofline, and matrix units cannot help a matrix-vector product). For M>1 it leaves roughly 9x on the table.

## The actual next build

A batched quant matmul using simdgroup_matrix: dequantize a weight tile into threadgroup memory in 8x8 blocks, load it and the activation tile as simdgroup_float8x8, and accumulate with simdgroup_multiply_accumulate. The dequantization cost is then amortized across the tile instead of repeated per output element, and the multiply-accumulate moves off the scalar ALU.

This is a genuine new kernel rather than a tweak, and it is the first place in this codebase where the matrix units would be used at all. Scope it to Q4_K first — it is the format TinyLlama Q4_K_M leans on and the one with existing parity coverage at M>1 — and gate it on M above some threshold so decode keeps the cooperative path unchanged.

## Standing

  decode  : parity with llama.cpp (0.888x, inside its 19% run-to-run spread), bandwidth-bound at 92% of the memory roofline — essentially finished
  prefill : 5.2x behind, compute-bound at 11% of FLOP peak — roughly 9x of headroom, addressable only via the matrix units

## R-01M022K6X1ENYBCN7YVGXRV53M First matrix-unit quant kernel: bit-exact but 12.5x too slow — BM=8 gives no dequant amortization, and a capability gate turned a compile failure into a silent fallback
kind: research
state: draft
created: 2026-08-15

Built the first matrix-unit kernel in this codebase. It is correct and 12.5x too slow. Shipped disabled, with the diagnosis of exactly which design decision is wrong.

## Result

qmatmul_q4k_mm dequantizes an 8x256 weight tile into threadgroup memory and consumes it with simdgroup_multiply_accumulate.

Correctness: BIT-EXACT (0.000e+00 relative) against the scalar reference across six shapes, including partial tiles in both dimensions (M=9 N=13, M=33 N=40). Non-vacuity confirmed by mutation — perturbing the dequantized value turns the test red.

Performance:

| M | cooperative | matrix unit | ratio | GFLOP/s |
|---|---|---|---|---|
| 8 | 247.3us | 3094.3us | 0.08x | 60 |
| 64 | 1954.3us | 23541.2us | 0.08x | 63 |
| 256 | 7809.3us | 93808.7us | 0.08x | 63 |

0.9% of the 6.8 TFLOP/s peak, against the cooperative path's 11.1%.

## The design error

BM=8 query rows per threadgroup. The dequantized weight tile is therefore rebuilt once per QUERY TILE: at M=64 each weight is dequantized 8 times — exactly the amortization the cooperative kernel already achieves by looping rows — while ADDING X staging, two threadgroup barriers per super-block, and only 32 threads to do all of it.

The arithmetic is stark: 2048 dequantizations plus 4096 staged floats to feed 32 matrix MACs. The matrix units idle behind the staging. Using them bought nothing because the kernel never reaches them in any quantity.

What a working version needs: 128-256 threads per threadgroup rather than 32, and a much larger BM so ONE dequantized weight tile feeds MANY query sub-tiles. That is the whole point of a matrix-unit GEMM and this tiling does not do it. It is a redesign, not a bug fix.

Retained behind SetQ4KMatrixUnit with the parity test, so the redesign starts from a validated kernel rather than a blank file.

## Two probe findings that inspection would not have caught

**The kernel silently did not compile.** simdgroup_multiply_accumulate takes FOUR arguments (dst = a*b + c), not three. ensure_q4k_mm returned -10 from inside an `&&` chain and the dispatch fell back to the cooperative path with no error surfaced anywhere. The parity test PASSED at that point, with all-zero differences, because it was comparing the cooperative kernel against itself. Only a mutation probe — zero the kernel's output, expect red — exposed it, and only an fprintf inside ensure_ produced the compiler diagnostic.

Generalization: a capability gate written as `feature_available() && ensure_thing() == 0` converts a COMPILE FAILURE into a silent fallback. Any such gate needs either a probe that asserts the path is taken, or a loud failure when the resource it guards cannot be built.

**The first parity comparison used the wrong reference.** Against the COOPERATIVE kernel the new kernel showed 1.5e-4 to 1.2e-3 relative — which reads as a correctness bug. It is not. The cooperative kernel factors the min term out as d*sc*sum(x*q) - dmin*m*sum(x); this kernel and the scalar reference both compute x*(d*sc*q - dmin*m). Those are algebraically equal and round differently. Against the scalar reference the agreement is exact.

So a parity test can fail for the reason that its COMPARAND is a refactoring of the formula rather than the formula. Choose the reference whose arithmetic form matches.

## Standing, unchanged

  decode  : tg64 175.0 vs llama.cpp 190.4 = 0.919x — parity, bandwidth-bound at 92% of the memory roofline
  prefill : pp64 345.0 vs 1778.75 = 0.194x — 5.2x behind, compute-bound at 11% of FLOP peak

The prefill headroom is still there and still reachable only through the matrix units. This round established that the mechanism works and produces exact results; the next needs the tiling that actually feeds it.

## R-01M022X2PSEVSTDAWV34A87K59 Matrix-unit tile redesign: 9.3x but still 0.77x of the scalar path — 240 scalar dequantization instructions per matrix instruction rules out the approach, not just the tile
kind: research
state: draft
created: 2026-08-15

Redesigned the matrix-unit tile as the previous record specified. It gained 9.3x and still loses to the scalar path — and the measurement that explains why also rules out the whole approach, not just this tile.

## Result

8x8 tile on 32 threads -> 32x32 tile on 128 threads (4 simdgroups), so one dequantized weight tile feeds 32 query rows instead of 8.

  63 -> 585 GFLOP/s on the kernel, 9.3x

Still slower than the cooperative path at every batch size:

| M | cooperative | matrix unit | ratio |
|---|---|---|---|
| 8 | 246.0us | 1560.4us | 0.16x |
| 32 | 978.1us | 1761.2us | 0.56x |
| 64 | 1953.8us | 2845.6us | 0.69x |
| 256 | 7809.2us | 10092.0us | 0.77x |

The ratio rises with M and asymptotes below 1.

## Why, and why it is not a tuning problem

Per 64-wide K chunk a threadgroup dequantizes 2048 weight values at roughly 15 integer/float operations each — about 30720 scalar operations — to feed 128 simdgroup MAC instructions.

**240 scalar instructions per matrix instruction.**

The matrix units are idle behind the dequantization. No tile shape repairs a 240:1 ratio: raising BM improves FLOP-per-dequantization only linearly (BM=32 gives 64, BM=128 gives 256) while threadgroup memory caps BM long before the ratio inverts. This is the fact that generalizes beyond the kernel: on 4-bit weights the multiply is not the work — the unpacking is.

And the cooperative path does the SAME dequantization while reaching 757 GFLOP/s, precisely because it never stages: dequantize in registers, multiply immediately. Adding threadgroup staging plus two barriers per chunk to an already dequantization-bound kernel cannot win, whatever the tile.

## What this rules in

Not a better tile — removing dequantization from the inner loop. Dequantize a layer's weight to f16 ONCE per prefill (~24 MB -> ~48 MB transient per layer) and hand the batch to an optimized f16 GEMM. The dequantization is then paid once per PROMPT rather than once per query tile, and the GEMM runs on hardware paths that are already near peak.

That also matches the shape of llama.cpp's advantage: pp64 1778 tok/s is far above anything a per-tile-dequantizing kernel reaches, and MPS f16 GEMM is already available in this codebase for the f32 path.

Cost note: the transient f16 buffer is per LAYER, not per model — 22 layers are processed in sequence, so peak extra memory is one layer's dequantized weight, not the whole 636 MB model.

## Process

Correctness held across the redesign: bit-exact (0.000e+00) against the scalar reference on all six shapes including partial tiles in both dimensions, with a mutation probe confirming the redesigned kernel is the one under test.

ensure_q4k_mm now PRINTS the MSL build error rather than returning -10 into an && chain. Last round that silently hid a kernel that never compiled, and every measurement taken before the probe was of the fallback path. The fix is cheap and permanent.

## Standing, unchanged

  decode  : tg64 175.0 vs llama.cpp 190.4 = 0.919x — parity
  prefill : pp64 345.0 vs 1778.75 = 0.194x — 5.2x behind

## R-01M0235NX0ED08G20HT3CEHNGZ Dequantize-once + dense MPS GEMM beats the quant kernel above M=48, 3.86x at M=256 — the mechanism that closes prefill, with the transpose store as the quantified next fix
kind: research
state: draft
created: 2026-08-15

The direction the matrix-unit work ruled in is confirmed: taking dequantization OUT of the inner loop beats the quant kernel at prefill batch sizes, 3.86x at M=256.

## Result

dequant_q4k_t expands a resident Q4_K weight once into a dense f32 [K][N] buffer; a dense MPS GEMM then runs the batch. K=2048, N=5632, GPU timestamps:

| M | cooperative | dequant+GEMM | ratio |
|---|---|---|---|
| 8 | 246.6us | 1122.7us | 0.22x |
| 32 | 978.7us | 1127.8us | 0.87x |
| 64 | 1954.4us | 1360.2us | 1.44x |
| 128 | 3907.3us | 1622.2us | 2.41x |
| 256 | 7809.4us | 2138.9us | 3.65x |

Crossover near M=48.

## Why it works where the matrix-unit kernel did not

Both approaches use the same hardware. The difference is what the inner loop contains.

MPS GEMM reaches 3207 GFLOP/s at M=64 and 4486 at M=256 — 47% and 66% of the 6.8 TFLOP/s peak — against the quant kernel's 757 GFLOP/s (11%) and the hand-written matrix-unit kernel's 585 (8.6%). A dense GEMM has no dequantization in its inner loop at all, so its instruction budget is arithmetic. The fixed cost of expanding the weight is then paid ONCE per matmul instead of once per query tile, and above the crossover the GEMM's 4-6x better efficiency more than covers it.

This also explains the earlier failure precisely rather than by analogy: the matrix-unit kernel was not slow because of tile shape (the redesign proved that, 9.3x and still losing) but because dequantization in an inner loop is a 240:1 instruction ratio no tiling can invert. Moving it out is the only structural fix, and moving it out means the multiply can be an off-the-shelf GEMM.

## Quantified weakness

The dequantization is the slow half: 814us to write 44 MB is 57 GB/s, roughly a third of what this machine sustains elsewhere. The cause is the transposing store Out[(k0+l)*N+n], which strides by N and does not coalesce — the source layout is [N][K] and MPS wants [K][N]. A staged-tile transpose through threadgroup memory should reach ~180 GB/s, cutting about 560us from every row of the table and moving the crossover below M=32.

That is worth doing before wiring, because it changes which prompt lengths benefit.

## Deliberately not wired

Routing prefill through this needs a scratch-buffer strategy (44 MB per projection, transient, reusable across layers since they run in sequence) and a crossover threshold. That is its own unit with its own end-to-end validation, and wiring it now on top of an un-tuned dequantization would set the threshold from a number about to change.

## Standing

  decode  : tg64 175.0 vs llama.cpp 190.4 = 0.919x — parity, bandwidth-bound at 92% of the memory roofline
  prefill : pp64 345.0 vs 1778.75 = 0.194x — 5.2x behind; this is the first mechanism measured to close a real part of it

Order of remaining work: (1) coalesce the dequantization transpose, (2) re-measure the crossover, (3) wire prefill through it above the threshold with a reusable scratch buffer, (4) re-run the pp64 head-to-head.

## R-01M023GH4YFVJVYK66CNVQZS4K Prefill wired through dequant+GEMM: coalescing the transpose gave 3.1x on the expansion, crossover fell to M~18, pp64 gap 5.2x -> 3.0x
kind: research
state: draft
created: 2026-08-15

Steps 1-4 of the recorded plan executed. Prompt processing is 1.7-1.8x faster end to end and the llama.cpp gap narrows from 5.2x to 3.0x.

## 1. Coalescing the transpose

Writing Out[(k0+l)*N+n] from the thread owning sub-block (n,s) strides by N between consecutive stores. Staging the tile through threadgroup memory so 32 threads write 32 CONSECUTIVE n per step:

  814.0 -> 259.6us for 44 MB, 57 -> 178 GB/s, 3.1x

178 GB/s is close to this machine's ~200 GB/s roofline, so the store side is now essentially done.

The refactor is numerically neutral, and that was checked rather than assumed: the parity figures against the quant kernel are unchanged to three digits (8.09e-05 at M=64, 2.86e-04 at M=256). A transpose that changed results would have shown here.

## 2. Re-measured crossover

| M | cooperative | dequant+GEMM | ratio |
|---|---|---|---|
| 8 | 305.6us | 553.5us | 0.55x |
| 16 | 490.5us | 542.2us | 0.90x |
| 24 | 734.4us | 548.0us | 1.34x |
| 32 | 978.4us | 540.5us | 1.81x |
| 64 | 1954.1us | 762.7us | 2.56x |
| 256 | 7809.4us | 1453.9us | 5.37x |

Crossover moved from M~48 to M~18. Note the fused path is nearly FLAT from M=8 to M=48 (553 -> 602us) because the fixed dequantization dominates there and the GEMM is almost free by comparison; the quant kernel meanwhile scales linearly in M. That shape is why the crossover is sharp.

## 3. Wiring

Routed at M>=24, a margin over the measured crossover. The dequantized weight goes into one lazily-grown scratch buffer reused by every projection and every layer — safe because compute encoders in a command buffer execute in submission order, so dequant(A)->gemm(A)->dequant(B)->gemm(B) cannot race. Peak extra memory is one projection's expansion (~92 MB for the column-fused gate|up), not the model.

## 4. End to end

  pp64  359.6 -> 662.7 tok/s isolated (1.84x); 585.8 in the head-to-head run
  tg64  unchanged — decode is M=1 and never reaches the threshold

Against llama.cpp pp64 1778.75: 0.194x -> 0.329x. Prompt-processing gap 5.2x -> 3.0x.

Correctness: generated token streams IDENTICAL with the path on and off, over 32 generated tokens from a 32-token prompt so prefill runs at M=32 and takes the new route.

## What the remaining 3x is

The dequantization, and it is now bandwidth-bound rather than wasteful: 44 MB written per projection at 178 GB/s against a ~200 GB/s ceiling. It cannot get materially cheaper in f32.

Expanding to f16 instead halves the traffic to 22 MB — roughly 130us instead of 260 — and MPS f16 GEMM is typically faster than f32 on this hardware, so the gain compounds. Q4_K carries ~4 significant bits per weight and the scales are already f16 in the block header, so f16 expansion loses nothing the format holds; that claim needs the same on/off token-identity check this round used before it is believed.

Order for next: (1) f16 expansion + f16 GEMM, (2) verify token identity and re-measure, (3) if the gap persists, profile whether the residual is the GEMM's own efficiency or the expansion.

## R-01M023RDTKFZ89AFWB5V54DAPC f16 GEMM is only ~1.07x, and weight expansion alone is 63% of llama.cpp's per-layer budget — the expansion approach tops out ~2.4x short, so the quantized-GEMM conclusion was too strong
kind: research
state: draft
created: 2026-08-15

Measured the f16 expansion idea before building it. It is worth about 1.27x, and the budget it exposes shows the whole expansion APPROACH has a ceiling ~2.4x short of llama.cpp. Recording that ceiling is the useful result.

## What was measured

MPS f16 GEMM against f32, same shapes:

| M | f32 | f16 | ratio |
|---|---|---|---|
| 32 | 264.0us (2796 GFLOP/s) | 236.0us (3128) | 1.12x |
| 64 | 449.3us (3286) | 317.0us (4657) | 1.42x |
| 128 | 741.8us (3981) | 723.5us (4081) | 1.03x |
| 256 | 1268.3us (4656) | 1189.9us (4963) | 1.07x |

Mostly ~1.07x, with one favourable point at M=64. So the f16 path's value is almost entirely halving the expansion write, not speeding the multiply.

## The budget that matters

  GoAI pp64 585.8 tok/s -> 109.3 ms for 64 tokens -> 4.97 ms/layer
  llama.cpp  1778.75    ->  36.0 ms              -> 1.64 ms/layer

  weight expansion alone: 4 projections x 260us = 1.04 ms/layer
                        = 63% of llama.cpp's ENTIRE per-layer time

Even a perfect f16 expansion (half the write, plus the small GEMM gain) saves ~1.05 ms/layer and lands at pp64 ~743 tok/s — still 2.4x behind.

**The expansion approach cannot beat an implementation that never expands.** llama.cpp keeps weights quantized and runs a quantized GEMM on the matrix units; its per-layer budget has no expansion term at all. Everything gained this round and last (102 -> 345 -> 586 tok/s) came from making expansion cheap, and that line of attack tops out around 743.

## Where that leaves the matrix-unit kernel

Closing the rest requires the thing that failed earlier: a quantized GEMM that dequantizes inside the kernel efficiently enough. My attempt reached 585 GFLOP/s and I concluded from a 240:1 scalar-to-matrix instruction ratio that the approach was closed. That conclusion was too strong. llama.cpp demonstrably does exactly this and is ~3x faster than GoAI's current best, so the ratio is not a property of the problem but of my kernel — most likely the dequantization arithmetic itself (bit-twiddling per element) rather than the tiling, since the tiling redesign already gave 9.3x without closing the gap.

The correction matters: "the matrix-unit approach is closed" should have been "my matrix-unit kernel spends too much on dequantization", which is a different and falsifiable claim.

## Not built

The f16 path is real but modest (1.27x, ceiling 743 tok/s) and needs device-side f32<->f16 conversion kernels plus an accuracy validation. My first accuracy harness was unsound — it packed f16 pairs into float32 slices, where denormal bit patterns passing through a float32 register can be flushed — so it produced a >100% error that says nothing about MPS. A real check needs a conversion KERNEL, not host bit-packing.

Reverted the unused MatMulF16 rather than leave unvalidated code in the tree.

## Standing

  decode  : tg64 0.914x — parity
  prefill : pp64 0.329x — 3.0x behind, was 5.2x at the start of this line of work

Next, in order of expected value: (1) profile the dequantization arithmetic inside a matrix-unit kernel to find where the 240:1 goes, since llama.cpp proves it can be much lower; (2) f16 expansion as the incremental 1.27x if (1) stalls.

## R-01M02431SFE9VS94RN2V14AM0P Hoisting per-sub-block constants gave 2.35x and the matrix-unit kernel now beats the scalar path — retracting the '240:1 rules out the approach' conclusion
kind: research
state: draft
created: 2026-08-15

Profiled where the dequantization instructions went, as the previous record specified. Most of the 240:1 ratio was redundant work, not inherent cost — which retracts the conclusion I drew from it.

## The finding

The matrix-unit kernel's staging loop recomputed, for EVERY weight: the base address, d, dmin, and the 6-bit scale/min unpack. About 15 operations per element, of which 12 are identical across the 32 elements sharing a Q4_K sub-block.

Giving each thread one (n, 16-element run) pair and computing those constants once:

  585 -> 1377 GFLOP/s, 2.35x

16 divides 32, so a run never straddles a sub-block boundary and hf is constant within it — which is what makes the hoist legal.

## Honest ranking, K=2048 N=5632

| M | cooperative | matrix unit | dequant+GEMM |
|---|---|---|---|
| 32 | 979.7us | 719.5us | 526.3us |
| 64 | 1955.1us | 1198.1us | 743.0us |
| 128 | 3905.5us | 2214.2us | 1017.4us |
| 256 | 7807.8us | 4288.4us | 1558.5us |

The matrix-unit kernel now BEATS the scalar cooperative path by 1.36x-1.82x. It still loses to dequantize-once + dense GEMM by ~2.75x, so it stays disabled.

## The retraction

I previously concluded that a 240:1 scalar-to-matrix instruction ratio "rules out the approach, not just the tile". That was wrong, and wrong in an identifiable way: I measured the ratio and treated it as a property of dequantizing-inside-a-kernel, when most of it was redundant per-element scale work that any careful implementation would hoist. The claim should have been "my kernel's dequantization is doing 4x more work than necessary", which is falsifiable and turned out to be true.

The chain is worth recording as a whole: tile redesign 9.3x (real, diagnosis fine), then "approach closed" (wrong), then hoisting 2.35x (real, and only attempted because llama.cpp's existence contradicted the closure claim). The external evidence — a competitor doing the thing I had declared impossible — was what forced the re-examination. Without it the wrong conclusion would have stood.

## Why dq+gemm still wins

What remains in the matrix-unit kernel is the staging: every weight written to threadgroup memory and read back, with two barriers per K-chunk. dq+gemm writes once to device memory and hands the multiply to a tuned MPS kernel that has no dequantization in its inner loop at all.

That gap is not obviously closable by more tuning of my kernel, and the expansion approach has its own measured ceiling (~743 tok/s pp64, 2.4x short of llama.cpp). Both roads are now mapped with numbers.

## Measurement discipline

All three arms are pinned explicitly now. QMatMulResident routes by flags, so an unpinned arm measures whichever path the flags select — precisely how the crossover benchmark came to compare dq+gemm against itself last round.

## Standing

  decode  : tg64 0.914x — parity
  prefill : pp64 0.329x — 3.0x behind

## R-01M024DQCHEXH8PT2TATEE04H0 f16 prompt-processing chain: 1.08-1.39x, produces NaN from f16 range overflow, and the accuracy harness reported zero error because NaN comparisons are false
kind: research
state: draft
created: 2026-08-15

Built the f16 prompt-processing chain, measured it, and reverted it. The gain is smaller than estimated and it overflows f16 range — which my accuracy harness reported as ZERO error because of how NaN compares.

## What was built and measured

Full chain: convert A to half, expand Q4_K straight to half, f16 MPS GEMM, convert the result back. Against the f32 chain, K=2048 N=5632:

| M | f32 chain | f16 chain | ratio |
|---|---|---|---|
| 32 | 531.0us | 446.6us | 1.19x |
| 64 | 731.8us | 524.9us | 1.39x |
| 128 | 1014.6us | 910.9us | 1.11x |
| 256 | 1550.6us | 1442.2us | 1.08x |

1.08-1.39x, against the 1.66x predicted from component times. The prediction added dequant 130 + GEMM 317 = 447us and ignored the two conversion passes and the extra encoders, which is most of the shortfall.

## The failure, and the harness bug that hid it

The f16 chain produces NaN. Reference output values reach -5.68e6; f16's maximum finite value is 65504. The result overflows to infinity and the conversion back yields NaN.

The accuracy check reported maxRel = 0.00e+00 at EVERY M. The loop was the usual form:

  d := abs(got - ref); if rr := d/den; rr > maxRel { maxRel = rr }

With got = NaN, d is NaN, and `NaN > maxRel` is FALSE. A max-tracking comparison silently ignores NaN, so a completely broken result reports perfect agreement.

**Any max/min-tracking accuracy loop needs an explicit NaN check.** This is the second unsound f16 harness in two rounds — the first packed f16 pairs into float32 slices where denormals can be flushed. Both produced confident numbers that meant nothing.

What exposed it was the mutation probe: perturbing the dequantized weight by 5% changed nothing, which is impossible for a live path. The probe said "not reached", the ensure-probe said "reached", and only dumping the actual values resolved the contradiction. Two probes disagreeing was the signal that neither answer was the real one.

## Judgement

Reverted. 1.08-1.39x does not justify a path that needs range analysis per model to be safe: the overflow is data-dependent, so it could pass every synthetic test and fail on a real activation distribution. The value here is the measurement, not the code.

Worth noting the overflow may be specific to this test's random-byte weights, which produce far larger products than a trained model would. A version that keeps the ACCUMULATION and OUTPUT in f32 while storing only the weight in f16 would avoid it entirely — MPS requires matching operand types, so that needs a different multiply than MPSMatrixMultiplication.

## Standing, unchanged

  decode  : tg64 0.914x — parity
  prefill : pp64 0.329x — 3.0x behind

Both remaining roads are now measured end to end: expansion tops out around 743 tok/s pp64 (2.4x short), and the matrix-unit kernel needs its threadgroup staging removed to go further. Neither is a small change, and the f16 shortcut between them is closed.

## R-01M024MQKEE249NDV49JH678EH Reverted the A-staging removal — the staging zero-pads partial M tiles and a clamp misattributes rows; both prefill roads are now measured to their limits
kind: research
state: draft
created: 2026-08-15

Attempted the staging removal the previous record specified. It broke partial-M correctness, the parity test caught it, and the expected payoff would only have reached parity with the path already in production. Reverted.

## What was attempted

Read the A tile straight from device memory instead of staging it in sX — freeing 16 KB of threadgroup memory and one barrier per K-chunk — and raise BM from 32 to 64 so the weight tile serves twice as many query rows.

## Why it does not work as stated

The sX staging is not redundant: it ZERO-PADS partial M tiles. Loading A directly requires clamping the row index to stay in bounds, and a clamp silently misattributes rows. At M=9 the second simdgroup reads rows 1..8 while storing to output rows 8..15, so output row 8 receives row 1's result.

TestQ4KMatrixUnitMatchesCooperative caught it at exactly that shape: max relative 1.374e+01 at M=9 K=256 N=13, while every full-tile shape stayed at 0.000e+00.

That test was built with partial tiles in BOTH dimensions specifically because this class of bug is invisible to inspection. The clamp reads as a bounds fix, and it is one — it just answers a different question than the one that matters. Without the M=9 case the change would have looked correct and shipped.

## Why it was not fixed

A dual path (direct load when mi0+BM<=M, staged otherwise) would work. But the whole change is worth at most ~1.5x on a kernel currently ~2.75x behind dq+gemm, so the best case is parity with the path already in production. Not worth the added branch and the second correctness surface.

## Where both roads now stand, measured

Production (dq+gemm), M=64: 743us total = expansion 260us + GEMM 449us.
  - expansion writes 44 MB at 178 GB/s, against a ~200 GB/s roofline
  - GEMM runs at 3286 GFLOP/s, 48% of the 6.8 TFLOP/s peak
  Both halves are near their respective limits. There is no significant headroom left in this path.

Matrix-unit kernel, M=64: 1198us at 1232 GFLOP/s (18% of peak). Reads only the 6.5 MB quantized weight, so its ceiling is far higher — but the remaining cost is threadgroup staging, and removing the part that is safe to remove (sW) is not possible: the matrix units need their operands in memory.

f16 shortcut: closed (1.08-1.39x, and f16 range overflow producing NaN).

## Honest assessment of the remaining gap

pp64 is 3.0x behind llama.cpp and every incremental lever measured this session is now spent or bounded below what would close it. Closing it means matching a hand-tuned quantized GEMM — the same kernel class llama.cpp ships — rather than any further adjustment to what exists here. That is a substantial engineering effort, not a tuning pass, and it should be scoped as such rather than approached in increments.

  decode  : tg64 0.914x — parity, bandwidth-bound at 92% of the memory roofline
  prefill : pp64 0.329x — 3.0x behind, from 0.057x when this line of work began

## R-01M024SWT9EJBRZSJJWJ5GCS16 Prefill budget does not add up: ~1.5-2.3 ms/layer unaccounted, larger than the expansion cost — and the GPU-time instrument does not cover the prefill submission path
kind: research
state: draft
created: 2026-08-15

Checked whether the prefill budget adds up before doing more kernel work. It does not, and the shortfall is LARGER than the expansion cost the last several rounds went into optimizing.

## The accounting

Per layer at M=64, from measured component rates:

  expansion  0.99 ms   (4 projections, 178 GB/s measured)
  GEMM       1.72 ms   (5.64 GFLOP at the measured 3286 GFLOP/s)
  sum        2.71 ms

  measured   4.16 ms/layer (91.5 ms for 64 tokens, best of 5)
  UNACCOUNTED ~1.45 ms/layer

At the slower 585.8 tok/s figure from the head-to-head the shortfall is 2.26 ms/layer. Either way it is comparable to or larger than the expansion.

## Why this matters for the strategy

llama.cpp does the same 5.64 GFLOP/layer in 1.64 ms = 3.45 TFLOP/s — essentially the SAME rate as MPS f32 GEMM (3286 GFLOP/s). Its advantage is not a faster multiply. It is that its per-layer time is almost exactly its GEMM time: no expansion, and no unaccounted remainder.

So GoAI's prefill gap decomposes as roughly: expansion 0.99 ms (removable only by a quantized GEMM) plus an unattributed 1.45-2.26 ms (unknown). The last several rounds treated the expansion as the whole remaining problem. It is at most half.

## What the profile does NOT show

Per-kernel element counts across a full 64-token prefill are all small — QMatMulResident 25.3 Melem, BinaryN 13.7, RMSNorm 5.8, RoPEPair 2.0, Copy2D 2.0, Blit 0.3, MHAAt 0.09. No oversized operation of the kind that explained the decode case. Total dispatch count is ~333 for all 22 layers, so per-dispatch overhead at the measured ~2 us marginal is ~0.7 ms for the WHOLE prefill, not per layer.

So the remainder is not a bloated op and not dispatch count. The candidates left are the GEMM running slower at the ACTUAL projection shapes than at the K=2048 N=5632 shape I measured it on, attention at sq=64, or host-side time between submissions.

## Blocked on instrumentation

LastGPUSeconds returned 0.00 for prefill: it reads the last buffer waited through mtl_recorder_wait, and the prefill path evidently does not submit through it. Attribution needs that instrument extended to whatever path StepNLast uses. That is the next concrete step, and it is small.

Recording this rather than guessing, because the same shortcut — estimating a component rather than measuring it in situ — is what made the elementwise-over-the-whole-context bug invisible for several rounds, and what made the expansion look like the whole story here.

## Standing

  decode  : tg64 0.914x — parity
  prefill : pp64 0.329x — 3.0x behind; roughly half the remaining gap is now known to be something other than weight expansion

## R-01M025E3E8ETVANGJ2TXK7AD5Y Q4_K_M is a MIXED file: ffn_down is Q6_K on 10 of 22 layers and fell off the prefill fast path — routing it gives 1.10x, and an exact weight-level test settles what the matmul-level one cannot
kind: research
state: draft
created: 2026-08-15

The prefill budget not adding up led to a mixed-format blind spot: Q4_K_M is not all Q4_K, and the largest projection in half the layers was falling off the fast path.

## The chain

The previous record flagged ~1.45 ms/layer unaccounted. Extending the GPU-time capture to mtl_recorder_finish — prefill submits through THAT, not commit+wait, which is why LastGPUSeconds read 0.00 — gave the split:

  n= 64: wall 91.57 ms  gpuBusy 89.40 ms  gpu 97.6%  host 2.18 ms  4.06 ms/layer
  n=128: wall 141.88 ms gpuBusy 138.84 ms gpu 97.9%  host 3.04 ms  6.31 ms/layer

So it is GPU work, not host. And it scaled with M, which rules out fixed overhead.

Next candidate was my GEMM rate, measured at ONE shape (K=2048 N=5632). Re-measured at the four real projection shapes: 1.70 ms/layer at M=64 against the 1.72 estimated. The estimate was right; the GEMM is not the problem.

That left "some projections are not on the fast path". Reading the tensor types out of the GGUF:

  type 0  (F32)  x45   norms
  type 12 (Q4_K) x135
  type 14 (Q6_K) x21   = attn_v x10, ffn_down x10, output.weight

ffn_down is the largest projection in a layer, and it is Q6_K on 10 of 22 layers. The prompt-processing path was gated on qt==12.

## Fix and result

dequant_q6k_t, same staged-tile transpose as the Q4_K variant.

  pp64  698.1 / 699.9  ->  766.9 / 770.6 tok/s   1.10x, interleaved
  tokens identical over 32 generated from a 32-token prompt

## Correctness, and choosing a test that can settle it

The matmul-level parity test shows 1.5e-4 (K=256) to 9.5e-4 (K=2048) against the SCALAR kernel — and that is the same order whether the reference is the scalar or the cooperative kernel, so it is accumulation order (MPS versus a sequential dot product), not the regrouping I first suspected.

But a test with that noise floor cannot distinguish a wrong dequant from reassociation. TestQ6KDequantExact compares the expanded WEIGHTS against a CPU decode of the ggml block layout: 0.000e+00, exact. The matmul test then guards wiring and shapes at 1.5e-3, and the exact test guards the arithmetic.

The general point: when a parity test's noise floor exceeds the error a bug would produce, add a test at a level where the answer is exact. Do not loosen the threshold and call it passing.

## Two harness traps hit

A pseudo-random byte fill puts the Q6_K block scale (a half at offset 208) into the NaN/Inf exponent range for some blocks. BOTH paths then produce NaN, and the comparison reports agreement — the same NaN-comparison hazard that made the f16 harness report zero error last round. Second occurrence in two rounds; the tests now write a valid scale explicitly.

A shell string comparison of captured multi-line output reported DIVERGED when the token streams were in fact identical. Comparing parsed integers said 0 of 32 differing. Structured comparison, not string equality.

## Standing

  decode  : tg64 174.84 vs llama.cpp 171.95 = 1.017x  (inside its ~19% run-to-run spread; parity)
  prefill : pp64 598.3 vs 1778.75 = 0.336x

Still open: ~1.4 ms/layer of GPU work unattributed after this fix. Per-kernel element counts are all small and total dispatches are ~333 for the whole prefill, so it is neither a bloated op nor dispatch overhead. The next measurement should time the individual encoders inside one prefill command buffer rather than infer from component rates — the inference has now been wrong twice.

## R-01M025MS9ZEVRVCJ3BBNHZSWQ0 Prefill budget closes to 85%: attention is 84% of non-matmul GPU work at 3.1% of peak — small now, structurally wrong for long prompts because causal routes to the decode kernel
kind: research
state: draft
created: 2026-08-15

The prefill budget now closes to 85%, and the last unexplained component turned out to be attention — small today, structurally wrong for long prompts.

## Closing the budget

Every per-op figure the budget used had been measured at DECODE shape (rows=1). Prefill runs at rows=64. Re-measured there, per layer:

  2 x RMSNorm           12.6 us
  SwiGLU 64x5632         9.1 us
  2 x residual add       8.4 us
  attention sq=64 sk=64 159.3 us
  total                 189 us

  expansion 0.99 + GEMM 1.70 + non-matmul 0.54 = 3.23 ms/layer
  measured 3.79 ms/layer -> 15% unaccounted, down from 34%

That is the third time this session the fix was "measure the component in the regime it actually runs in" rather than anything cleverer. The pattern is now unmistakable: cache-vs-DRAM for the roofline, hidden-vs-serial for encode, decode-shape-vs-prefill-shape here.

## The finding

Attention is 84% of the non-matmul GPU work and runs at 211 GFLOP/s — 3.1% of the 6.8 TFLOP/s peak. Every other non-matmul op together is 30 us/layer.

It is NOT today's bottleneck: 159 us is ~4% of a 3.79 ms layer at a 64-token prompt. It is a latent one. Attention is O(sq*sk) and the measured points already trace the curve — sk=64 159us, sk=128 354us. At a 512-token prompt it would dominate the layer outright, and long prompts are exactly the case prompt processing exists for.

## Cause is structural

mtl_recorder_mha routes causal!=0 to the DECODE kernel. That kernel streams keys per query row: correct and near-optimal for sq=1, wrong for sq=512, where keys should be tiled through threadgroup memory and reused across query rows.

This is the SAME distinction that separated the cooperative matmul from expand-then-GEMM: a kernel shaped for one row applied to many rows re-reads what it should reuse. Prefill attention has the identical defect and it has not been touched because no measurement had ever looked at attention at sq>1.

Worth noting how it stayed hidden: the flash and two-pass attention kernels were both optimized earlier this session (4.6-6.4x from the runtime-loop-bound fix), but neither is what prefill uses — the causal route bypasses them.

## Standing

  decode  : tg64 174.84 vs llama.cpp 171.95 = 1.017x — parity
  prefill : pp64 598.3 vs 1778.75 = 0.336x

Remaining prefill gap by component, per layer: expansion 0.99 ms (removable only by a quantized GEMM), GEMM 1.70 ms (at 3311 GFLOP/s effective, near MPS's limit), attention 0.16 ms (structurally wrong but small at short prompts), 0.56 ms unattributed.

llama.cpp's entire layer is 1.64 ms. Its advantage remains that it has no expansion term, which is the same conclusion as three rounds ago — but the budget is now measured rather than assumed, so the next attempt can be scoped against real numbers.

## R-01M025TQDFF969T3S98R79T1W9 Neither attention kernel suits prefill: flash is 0.72-0.97x of the decode kernel and both run at 3-4% of peak — the reroute is unavailable, a real flash kernel is needed
kind: research
state: draft
created: 2026-08-15

Tested the obvious fix for prefill attention — reroute causal sq>1 from the decode kernel to the flash kernel — and it is not available. Both kernels are equally poor.

## Measurement

M2 Pro, heads=32 kvHeads=4 dk=64:

| seq | decode kernel | flash | ratio |
|---|---|---|---|
| 64 | 164.3us (204 GFLOP/s, 3.0% peak) | 228.0us | 0.72x |
| 128 | 521.4us (257 GFLOP/s, 3.8% peak) | 535.5us | 0.97x |
| 256 | 1902.3us (282 GFLOP/s, 4.2% peak) | 1987.5us | 0.96x |

They agree to 5.7e-05, so both are correct. The flash kernel — the TILED one, and the one this branch already sped up 4.7-5.4x with the runtime-loop-bound fix — is no faster here. Both sit at 3-4% of the 6.8 TFLOP/s peak.

## Reading it correctly

Both are quadratic, and attention must be. The scaling (x3.2 then x3.6 per doubling of seq) is not the defect. The CONSTANT is: 3-4% of peak where the matmul path reaches 48%.

Share of prefill by prompt length: ~4% at seq=64, 10% at seq=256 (41.9 ms of ~427 ms), and rising. That is why it stayed invisible — every measurement this session used a 64-token prompt, where attention is a rounding error, and the deficit only became legible once the rest of the budget was closed.

## Why the reroute fails

The flash kernel tiles K/V through threadgroup memory, which is the right structure. That it is still no faster suggests its tiling is not effective at these shapes — plausibly FTILE and the threadgroup geometry are tuned for a different regime, since the earlier 4.7-5.4x fix addressed its dequantization-style per-element cost, not its tiling.

Either way the conclusion is the same: closing this needs a genuine flash-attention kernel — Q tile resident, K/V streamed in tiles, online softmax — not a routing change. Both current kernels re-read K/V per query row in the shape that matters.

## State of the prefill programme

Every component is now measured, and the remaining work is two scoped projects rather than a series of adjustments:

  expansion  0.99 ms/layer  — removable only by a quantized GEMM (matrix-unit kernel is at 1377 GFLOP/s after hoisting, still 2.75x behind expand-then-GEMM)
  GEMM       1.70 ms/layer  — 3311 GFLOP/s effective, near MPS's limit, no headroom
  attention  0.16 ms/layer at seq=64 — 3-4% of peak, grows with prompt length
  unattributed 0.56 ms/layer

  decode  : tg64 1.017x — parity with llama.cpp
  prefill : pp64 0.336x — from 0.057x when this line began

llama.cpp's whole layer is 1.64 ms and has no expansion term. That remains the structural difference, and the two projects that would close it are now specified against measured baselines rather than guessed at.

## R-01M0262NX9E3SB6ED6YVK97AVT CORRECTION: prefill budget closes to 77%, not 85% — my non-matmul total double-counted an attention probe; every op is now measured and 23% remains, most likely dependency-chain serialization
kind: research
state: draft
created: 2026-08-15

Correcting my own previous record, and completing the prefill budget.

## The error

R-01M025MS9ZEVR reported non-matmul work as 0.54 ms/layer and the budget closing to 85%. That figure summed BOTH attention probes — including sk=128, which is not part of a 64-token layer. The real non-matmul total is 0.19 ms/layer, and the budget closes to 77%, not 85%.

The mistake was in the test itself: a running total accumulated inside a loop that measured two different sk values, and I read the printed subtotal without checking what it had added.

## The complete budget

Measured the remaining ops — RoPE 10.07us, Copy2D and Blit both immeasurably small, logits 329.6us once per prefill (0.015 ms/layer amortized):

| component | ms/layer |
|---|---|
| expansion | 0.990 |
| GEMM | 1.700 |
| attention | 0.159 |
| RMSNorm x2 | 0.013 |
| RoPE | 0.010 |
| SwiGLU | 0.009 |
| residual add x2 | 0.008 |
| logits (amortized) | 0.015 |
| TOTAL | 2.905 |

Measured 3.79 ms/layer -> 0.89 ms, 23%, unaccounted.

Every operation in the layer is now measured individually. The remainder is NOT a missing operation, which is what the last two rounds of searching assumed.

## The likely cause, and why it is familiar

Every figure above comes from repeating ONE op back-to-back on the same buffers. The GPU overlaps those iterations. A real layer is a dependency chain — norm feeds projection feeds attention feeds projection — with genuine data hazards between successive ops. A single-op microbenchmark structurally cannot show that serialization cost.

If that is right, it is the FIFTH instance this session of one error class: measuring in a regime the real workload does not run in.

  1. cache-resident weight vs DRAM streaming (roofline)
  2. hidden-behind-GPU vs serial (host encode)
  3. decode shape rows=1 vs prefill shape rows=64 (these ops)
  4. a path compared against itself (crossover benchmark, after wiring)
  5. isolated repeated op vs dependency chain (this)

The recurring shape is that a microbenchmark answers the question it was built to answer and not the one being asked, and the discrepancy only shows when a total is compared against the sum of its parts. That comparison has now caught four separate defects this session — the elementwise-over-whole-context bug, the Q6_K fast-path miss, the attention blind spot, and this.

## Testing the hypothesis

The check is direct: build one command buffer containing a real layer's op SEQUENCE on properly dependent buffers, and compare its GPU time against the sum of the same ops measured in isolation. If the chain is ~23% slower, that closes the budget and the conclusion is that per-op microbenchmarks systematically under-report by roughly that margin here.

## Standing, unchanged

  decode  : tg64 1.017x — parity
  prefill : pp64 0.336x

## R-01M0269TGAFSPR41BHC16JBYDH Dependency-chain and cache-reuse both disproven (1.01x, 1.00x) — the prefill gap is two extra projections per layer: q/k/v are separate while gate|up is fused
kind: research
state: draft
created: 2026-08-15

Tested the dependency-chain explanation for the 23% unaccounted prefill time. It is wrong, and so is the cache-reuse fallback. The cause is simpler and points at a concrete optimization.

## Both hypotheses disproven

A real layer's op sequence on dependent buffers, against the same ops timed in isolation:

  chain 2.986 ms/layer   isolated sum 2.965 ms   ratio 1.01x

No hazard penalty at all. Repeating the chain over EIGHT distinct weight sets, so successive iterations cannot reuse a cached weight:

  2.983 ms, ratio 1.00x

Worth noting why the second test was run: five earlier defects this session were all "measured in the wrong regime", so cache reuse was the obvious next suspect. It was not the cause. The pattern that had held five times did not hold a sixth, which is itself worth recording — a heuristic that keeps working invites assuming it always will.

## The actual cause

The synthetic layer does not reproduce the real dispatch COUNT. From the prefill profile:

  QMatMulResident  5.95/layer   (synthetic: 4)
  total dispatches ~15/layer    (synthetic: 11)

5.95 decomposes as q, k, v SEPARATE (3) + o (1) + gate|up fused (1) + down (1). So q/k/v are three separate projections while gate/up are column-fused — the opposite of what the synthetic assumed.

Total expanded BYTES are the same either way, so expansion cost is unchanged. But the GEMM is not: three small GEMMs replace one fused N=2560. The measured shape sweep shows why that costs — N=2560 runs at 2401 GFLOP/s while N=256 shapes sit far below, since small-N GEMMs are launch- and occupancy-limited rather than arithmetic-limited.

## Concrete candidate

Fusing q/k/v into a single projection. The weights arrive from GGUF as three separate tensors, and the fused code path ALREADY EXISTS (recordQKVProj records one projection through b.wqkv) — it is used for models that ship pre-concatenated qkv. Concatenating the three at load time would route TinyLlama down it, replacing three GEMMs and three expansions with one.

Expected: one 2048x2560 GEMM (279.6us measured) in place of q 2048x2048 (200.3us) plus two 2048x256, and two fewer dispatches and expansions per layer. Both the fusion machinery and the measurement to judge it already exist.

## Standing

  decode  : tg64 1.017x — parity
  prefill : pp64 0.336x

Budget per layer: expansion 0.99, GEMM 1.70, attention 0.159, everything else 0.055, measured 3.79 — with the 0.89 remainder now attributed to the two extra projections and the small-N GEMM penalty rather than to any modelled effect.

## R-01M026KP5BE6FSMP8K5BWG056R gate|up fusion implemented for Metal and REVERTED: consistently slower on prefill because it doubles the expansion scratch, despite standalone GEMM numbers predicting a win
kind: research
state: draft
created: 2026-08-15

Traced the extra projections to their exact cause, implemented the fix, measured it, and reverted it. The fusion does not pay on this backend.

## Where the 5.95 projections/layer come from

qfused bails on MIXED quant types:

  if q1.QT != q2.QT || q2.QT != q3.QT { return nil }   // -> unfused three-matmul chain

TinyLlama Q4_K_M has attn_v in Q6_K on 10 of 22 layers while attn_q/attn_k are Q4_K, so those 10 layers keep three separate projections. And qfused2 (gate|up) requires ops.fusedGateUp, which was CUDA-only.

The dispatch count settles it exactly:

  12 fused qkv + 10x3 unfused + 22 o + 44 gate/up + 22 down + 1 logits = 131   (measured 131)
  with gate|up fused it would be 109

So my earlier inference — "q, k, v are separate" — was half right: they are separate on 10 layers, and gate/up are separate on ALL of them.

## What was built

swiglu_halves for Metal (out[r,i] = silu(gu[r,i]) * gu[r,hidden+i]) plus the recorder method, enabling ops.fusedGateUp and routing the FFN through the single N=2*hidden projection.

## Why it was reverted

  unfused  pp 724.5 / 720.9   fused  pp 701.1 / 716.1
  unfused  tg 109.0 / 109.0   fused  tg 110.7 / 111.0

Prefill — the target — is consistently SLOWER fused. Decode is marginally faster, within noise. One generated token also differs, which is expected but not free: the fused N=11264 GEMM and two N=5632 GEMMs sum in different orders, and that can flip a greedy argmax on a near-tie.

The likely reason it loses: the expansion scratch. Fused needs one 92 MB expanded weight where unfused needs 46 MB twice. Same total bytes, but double the peak working set, and at these sizes that costs more than the GEMM shape gains. The standalone GEMM numbers that predicted a 231us/layer win (667us fused vs 2x449us) do not include the expansion, and expansion is now the dominant term in that path.

That is a specific instance of a general trap this session has hit repeatedly: a component measurement predicts a win that the system does not deliver, because the component was measured without the context that dominates it.

## Standing, unchanged

  decode  : tg64 1.017x — parity
  prefill : pp64 0.336x

The two extra-projection sources are now identified precisely. Fusing gate|up is measured and rejected. Fusing qkv on the 10 mixed-type layers would require expanding Q4_K and Q6_K into one buffer — the concatenation trick does not apply across quant types — which is a materially bigger change than qfused's byte append, and on this evidence would likely lose for the same scratch-size reason.

## R-01M026QJNSEE8TW9FRW8ETYSZQ Expansion scratch should be chunked to the largest tile that fits L2 (~22 MB): 1.13x measured, and it retroactively predicts the gate|up fusion loss
kind: research
state: draft
created: 2026-08-15

The gate|up fusion lost because it doubled the expansion scratch. Testing that directly gives a principled rule and a bounded next optimization.

## Measurement

Expand-then-GEMM for K=2048, N=5632, M=64, with the expansion split along N so the scratch is reused per chunk:

| chunks | scratch | total |
|---|---|---|
| 1 | 44.0 MB | 1.281 ms |
| 2 | 22.0 MB | 1.135 ms |
| 4 | 11.0 MB | 1.420 ms |
| 8 | 5.5 MB | 1.637 ms |
| 16 | 2.8 MB | 3.246 ms |

## What it means

Cache residency alone is NOT the explanation, which is what I expected going in. At 2.8 MB the expanded tile is certainly resident and it is 2.5x SLOWER, because N=352 GEMMs are launch- and occupancy-limited — the same small-N penalty the earlier dispatch-size sweep measured (N=2048 124 GB/s vs N=16384 170 GB/s).

The optimum is the LARGEST chunk that still fits L2. An M2 Pro has ~24 MB shared L2, and 22 MB wins while 44 MB does not. That is a rule with a machine constant in it, not a universal.

  chunk so the expanded tile is <= ~20 MB, and no smaller

## Applying it to the real projections

  qkv     K=2048 N= 2560  20.0 MB  -> 1 chunk (already fits)
  o       K=2048 N= 2048  16.0 MB  -> 1 chunk (already fits)
  gate|up K=2048 N=11264  88.0 MB  -> 4 chunks of N=2816
  down    K=5632 N= 2048  44.0 MB  -> 2 chunks of N=1024

Only two of the four projections would chunk at all. Taking the measured 1.13x on those two, and their share of the per-layer expansion+GEMM (gate|up and down are ~1.9 ms of the 2.69), the expected end-to-end effect is roughly 1.08x on prefill — real but modest.

This also retroactively explains the fusion result exactly: fusing gate|up made ONE 88 MB expansion where unfused made two 44 MB ones. Both exceed L2, but the fused one exceeds it by more, and the better GEMM shape did not compensate. The rule predicts that outcome, which is a check on the rule rather than a coincidence.

## Implementation cost

Chunking needs the GEMM to write a column range of C. MPSMatrixDescriptor takes rowBytes, so an MPSMatrix over the full C with rowBytes=N*4 and offset=n0*4 is a valid view of columns [n0, n0+CN) — the plumbing is an offset parameter on the matmul entry point, not a new kernel.

Not built this round. It is a bounded change with a measured 1.08x expectation, and it should be judged against the same end-to-end harness rather than the component number, since component numbers have predicted wins this session that the system did not deliver — the gate|up fusion being the most recent.

## Standing

  decode  : tg64 1.017x — parity
  prefill : pp64 0.336x

## R-01M027CXQ6EC99G4P3MSQAJQCR Chunked expansion implemented and REVERTED at 0.77x — the isolated 1.13x used contiguous per-chunk outputs, which the real strided-output version cannot have
kind: research
state: draft
created: 2026-08-15

Implemented the chunked expansion the previous record specified. It is a 0.77x REGRESSION end to end, and the isolated 1.13x that motivated it did not transfer for an identifiable reason.

## Result

Interleaved, on a machine verified quiet by the noise-floor guard:

  HEAD    723.1 / 714.8 / 733.9 tok/s
  chunked 560.0 / 536.7 / 559.3 tok/s     0.77x

Non-overlapping. Reverted.

## Why the isolated measurement did not transfer

The synthetic test that produced 1.13x gave every chunk its own CONTIGUOUS output buffer. The real implementation cannot: chunk n0 must write columns [n0, n0+cnt) of the shared C, which means an MPSMatrix with rowBytes = N*4 — a STRIDED output. Strided writes cost more than the cache-residency of the expanded tile saves.

So the validation test omitted the one cost the real version is forced to pay. That is the same error class as the gate|up fusion (component measured without the context that dominates it), and it is now the second consecutive optimization where an isolated number predicted a win the system did not deliver. The lesson is sharper than "measure end to end": the isolated test must model the CONSTRAINTS the real implementation cannot avoid, not just its arithmetic.

## A measurement-integrity incident worth recording

Midway through, a sequential HEAD-then-mine comparison read 751 vs 360 — an apparent 2x regression that sent me looking for a bug in the kernel diff, which is minimal and could not explain it. Then llama.cpp itself measured 108.79 tok/s against the 172-201 recorded earlier, and TestMeasurementNoiseFloor failed at cv 17.94% against its 12% threshold.

The machine had thermally degraded after a long session of continuous benchmarking. The guard built earlier this session for exactly this case caught it, and the correct reading was "no A/B is trustworthy right now" rather than anything about the code. After 90 seconds idle the guard passed again and the interleaved re-measurement gave the clean 0.77x above.

Two things that made this recoverable: the noise-floor guard existed, and llama.cpp was being measured LIVE rather than compared against a recorded constant — a stale constant would have shown GoAI collapsing to 0.55x of it and hidden the fact that the whole machine had slowed.

## Standing, unchanged

  decode  : tg64 ~1.017x — parity
  prefill : pp64 0.336x

The L2-fit rule from the previous record still describes the isolated behaviour correctly; it just cannot be exploited through a shared output buffer. Exploiting it would need the GEMM to write contiguously and a separate gather, which costs more than it saves.
