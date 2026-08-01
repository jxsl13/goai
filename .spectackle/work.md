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

## T-01KYJNDS5MFEDAR0GVMSNG4MHD Clear the remaining documentation debt to a green apicheck gate
kind: task
state: active
created: 2026-07-27
targets: internal/apicheck

Bring internal/apicheck to exit 0 so the public-API doc gate can be enabled.

Completed so far: 118 nlp struct-field symbols documented with architecture-aware godoc referencing the upstream tensor names (Bert, Gemma, Gemma2, Jamba, Mamba, Mamba2, Mixtral, DeepSeekV2, GraniteMoE, RWKV, T5, Qwen2MoE and the 16 quantized twins), 4 further symbols (classic GradientBoostingRegressor.Predict, safetensors TensorInfo.Name/Dtype/Shape), and the 3 magic backend-name string literals replaced with backend.CPU. Undocumented count went from 140 to 18; the magic-strings test is green.

Remaining: (a) the 18 undocumented symbols are all llamagpu New*Q8CUDA and New*Q4KCUDA constructors owned by the parallel CUDA worker; (b) the runnable-Example requirement of TestPublicAPIDocumentedWithExamples is a separate pass. Justified typeExampleExempt and methodExampleExempt allowlist entries are legitimate for fixture-heavy surfaces and beat boilerplate examples.

Definition of done: go test ./internal/apicheck exits 0, checked unpiped. Unblocks enabling apicheck in the CI always-run set.

Note: godoc and Example edits are .go files, so this consumes CI and belongs to the main agent rather than the docs lane.

Migrated from cavekit SPEC.md T892.

## T-01KYJNDSP0FG4B672MFS69AQ0F Enable apicheck and mdlint in the CI always-run set
kind: task
state: active
created: 2026-07-27
targets: internal/cichange

The source-walking meta-tests (internal/apicheck, internal/mdlint) walk the whole repo's source and markdown, and no import edge connects them to what they check, so import-graph impact selection never picks them and their invariants can rot red while CI stays green.

Mechanism is already built and live: a config alwaysRun field plus a repeatable -always-run flag; Impact() appends every configured package that exists in the graph to each non-empty selection. Docs-only and empty selections stay at zero runners. Missing packages are tolerated so temporary modules and renames cannot inject a bogus target. Unit tests plus the regression pinning that an nlp-only diff selects apicheck are in place and proven non-vacuous. internal/speccheck is already wired into the default always-run, proving the path end-to-end.

Blocked on: the default alwaysRun set is deliberately empty because enabling apicheck and mdlint while they are red on the committed tree would fail CI on the first push. apicheck is red on the remaining doc debt; mdlint is red on worker markdown.

When both gates are green, this is a one-line change.

Note: the mdlint blocker changes shape once the cavekit spec files are removed, since much of the red is in the generated spec views. Re-measure before flipping.

Migrated from cavekit SPEC.md T893.

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

## ADR-01KYQ9PHNPEFCVVMRXW6X22XN2 Titans Scan burns 24527 allocations per seq=128 forward but three attempts failed to make a fused path bit-identical (see R-01KYQ9CQ3XE1D) — how should it be resolved?
kind: adr
state: done
created: 2026-07-29
context: Established: the recurrence arithmetic is provably correct at seq=2/d=3; it is not input handling; not the matmul accumulation order or loop shape; the broadcast elementwise kernel is a plain scalar loop. At seq=5/d=8 exactly 3 of 64 memory elements differ by one ulp. PERF-FUSED-PATH-CHAIN-001 says a ~20-op chain needs a decision rather than another attempt.
decision: Scope down: keep matmuls and the outer product on the backend (bit-exact by construction), fuse only slices, transposes and elementwise — fewer allocations saved, no correctness risk
consequences: Decided without escalating: the alternatives are worse on their own terms, not merely less preferred. A tolerance gate would make Titans the only fused path in nn whose parity claim is weaker than its siblings' (GLA, DeltaNet, GatedDeltaNet, RGLRU, HGRN are all bit-exact), and it would trade a permanent, invisible correctness weakening for a bounded amount of speed. Continuing to chase FMA emission has already consumed three attempts against a one-ulp mismatch in 3 of 64 elements, with the three obvious causes eliminated; the expected cost of the fourth attempt is not visibly lower than the third. Dropping it forfeits the largest measured dispatch-overhead site outside backend/. Scoping down keeps every rounding decision inside the backend where it is already correct, which is why it carries no correctness risk at all rather than a small one. The cost is a smaller win: about 3 backend calls per timestep (two matmuls plus the outer product) instead of roughly eighteen, so the allocation count should fall by around an order of magnitude rather than to near zero. That is the right trade when the alternative is an unverifiable fast path. Re-measure rather than assume the reduction — the matmul outputs still allocate, and constructing the row tensors the matmuls need may claw back some of the saving; if it does, the honest outcome is to report the smaller number, not to reopen the bit-exactness question.
status: accepted

kind: radio
option: Scope down: keep matmuls and the outer product on the backend (bit-exact by construction), fuse only slices, transposes and elementwise — fewer allocations saved, no correctness risk
option: Accept a tolerance-gated fused path for Titans only, with an ADR recording why bit-exactness was given up
option: Drop it: leave Scan on dispatch and spend the effort on targets that validate bit-exactly
option: Keep chasing bit-exactness: instrument the exact FMA emission and match it
choice: Scope down: keep matmuls and the outer product on the backend (bit-exact by construction), fuse only slices, transposes and elementwise — fewer allocations saved, no correctness risk

## R-01KYQPKDVQFN3B8HMW3JBEC8JK Bounded-pool decode fan-out: throughput neutral, 4.5x fewer peak goroutines - and the nesting premise was false
kind: research
state: draft
created: 2026-07-29

MEASURED on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, min of 3 per arm, 3 alternations in one session (PROC-INTERLEAVE-001, PROC-BENCH-MINOFN-001).

CONTEXT. A rebase onto main put two parallelization mechanisms in format/gguf/quant_matmul.go side by side: main spawned GOMAXPROCS goroutines per call for the m==1 decode paths, this branch routed the m>1 general path through internal/parallel bounded pool (ADR-01KYMWJ76AFA2). Rather than leave one file with two policies, the decode helper was moved onto the pool and the change measured.

THROUGHPUT: a wash. Q8_0/Q4_K/Q6_K at M1, Q8_0/Q6_K at M16, Q8_0 M1 N4096, and a RunParallel batch benchmark all land inside noise (largest gap 3.6 percent, sign not stable across alternations). Nested prefill (BenchmarkQuantMamba2Prefill_256/_512, mixer parallel, QMatMul inside) also a wash: 1.9 percent one way at 256, 0.5 percent the other at 512.

RESOURCE USAGE: real and reproducible. Eight concurrent decode callers, sampled peak live goroutines - 22 through the pool, 98 and 101 spawning per call, across two alternations each. 4.5x. Aggregate throughput identical, so the bound is free.

THE FALSIFIED PREMISE, which is the durable part. The change was first justified by nesting: main recently parallelized the Mamba2 mixers (PR 577, 578), so QMatMul now runs inside a parallel region and a per-call spawn should give GOMAXPROCS squared goroutines. Measured: 37 peak either way. The reason is that prefill is m>1 and takes the general path, which was ALREADY pooled - the decode helper never runs nested there. The mechanism was right in the abstract and wrong about which code path reaches it. Writing the hypothesis into a code comment before measuring it would have shipped a false explanation attached to a true change.

BOTH PROBES SHIP as regression guards: TestGoroutinePeakConcurrentDecode and BenchmarkQMatMulQ8_0ConcurrentDecode in format/gguf.

GENERALIZABLE, and the reason no perfscan rule follows: the finding is that a goroutine-count claim needs a goroutine-count measurement, and that peak-live-goroutines is cheap to sample (a 20us ticker around the workload). That is a MEASUREMENT METHOD, not a source pattern - an AST scanner can see go inside a func but cannot tell whether that func has concurrent callers, which is the entire question. Flagging every per-call spawn would fire on correct code. Recorded here instead of forced into a rule (PERF-SCANRULE-EMPTY-001 reasoning: a rule that cannot separate the good case from the bad one has no signal).

## R-01KYQPMA40FVNRC6C38ZABN3DK Rebasing 180 perf commits onto a concurrently-optimized main: what the 16 conflicts taught
kind: research
state: draft
created: 2026-07-29

A 180-commit perf branch rebased onto a main that had been optimizing the SAME files in parallel. 16 conflicting commits. The resolutions split into five kinds, and the classification is the reusable part - blind hunk-picking is wrong in three of the five.

1. IDENTICAL INTENT, DIFFERENT MECHANISM. Both sides parallelized the same loop (QMatMul m>1; KNN Predict). Neither is more correct. Pick on a property that is not throughput - here, whether the fan-out is bounded - and then MEASURE the pick rather than asserting it (see R-01KYQPKDVQFN3, where the stated justification turned out false while the change stayed good).

2. SUPERSET vs SUBSET. Main had this branch scalar 4-way block PLUS a SIMD kernel. Take main wholesale.

3. DISJOINT IMPROVEMENTS TO THE SAME LINES. Main added a component jam, this branch added sample parallelism, both touching the same struct fields. Neither side alone is right and hand-merging six hunks blind is how a race gets shipped. Resolution: take main, drop this branch commit, re-book the work as its own measured task (T-01KYPCP21VFAZ).

4. ID COLLISION. Main claimed PS6005 for a different rule. The branch already contained its own renumber commit (PS6005 to PS6010) further down the todo - so the first resolution invented PS6014 and had to be undone. LESSON: before renumbering a colliding ID, look at the REST OF THE REBASE TODO, not just the current tree. A later commit already knows the answer.

5. FALSE CONFLICTS FROM AN EARLIER RESOLUTION. Once a both-appended tail is merged by hand, git re-flags the whole tail as new on every later commit that appends there - so the naive resolution silently re-adds 300 lines of already-present tests. Two edge cases: (a) both sides ending with the SAME closing lines makes git hoist them into the common tail, so concatenating ours+theirs leaves ours unterminated mid-function (caught by go vet: expected ( found TestDetect...); (b) the duplicate block must be diffed by DECLARATION NAME, not by size.

ORPHANS RUN BOTH WAYS (PROC-MERGE-ORPHAN-001). The known direction - take the other side wholesale, grep this side helper names for surviving call sites - fired once (parallelRows, fully dead, correctly deleted). The UNKNOWN direction bit harder: taking main gmm.go dropped parallelSamples, whose call site arrived cleanly in a LATER commit that had no conflict at all. So the grep must cover the whole remaining todo, not just the commit being resolved. Caught by go vet at the next stop.

GENERALIZABLE, and why no perfscan rule: every one of these is a VCS-state property (what the other side did, what the rest of the todo does). perfscan parses one file at one revision with go/ast and has no access to any of it. The mechanical parts are already covered - unused imports and undefined symbols by go vet, detached godoc by apicheck (PROC-GODOC-DETACH-002 fired again here, main stale helper doc landing above this branch new const). Recording as method.

## R-01KYQT329EEAD819VM5DGVSB0M REJECTED: contiguous back-substitution in linalg. Predicted 2-5x, measured 1.00x - the strided column stays cache-resident
kind: research
state: draft
created: 2026-07-29

REJECTS the central claim of T-01KYJQ3QRHFRS. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, min of 3-4 per arm, 3-4 interleaved alternations in one session.

THE CLAIM. All three triangular back-substitutions (LU.Solve linalg.go:126, CholSolve cholesky.go:82, Lstsq qr.go:107) read the already-solved tail as out[j*cols+c] with j innermost, striding by cols. At n=512 consecutive inner iterations sit 4KB apart, on a buffer of 2MB. Predicted 2-5x on the substitution phase and 1.6-3x on Inverse end to end, at medium-high confidence, mechanism called unambiguous.

MEASURED, layout change toggled in and out with the scratch hoist held constant in both arms so the ratio attributes only to the strided read:
  LUSolve 512x512   1.005x
  LUSolve 768x768   0.977x (baseline faster)
  Inverse n=512     0.994x (baseline faster)
  Lstsq n=512       1.001x
  CholSolve n=512   1.041x, consistent across four alternations
  CholSolve n=768   0.996x
One site, one size, four percent, gone at the next size up. Null.

WHY THE ANALYSIS WAS WRONG, which is the reusable part. The stride is real; the cache cost is not, because the analysis counted cache lines per READ and ignored REUSE. For a fixed column c, the back pass touches exactly the same n addresses (out[j*cols+c] for j in range) on every one of its n outer iterations. Those n lines total n*64 bytes of tag footprint - 32KB at n=512, 48KB at n=768 - and stay resident after the first pass. The working set is n LINES, not the 2MB buffer, so it never leaves L2 on this host and the fix has nothing to recover. A strided walk is only expensive when its footprint exceeds cache OR it is traversed once; this one is small AND re-traversed n/2 times.

CONTRAST with the strided walks that DID pay this session (NSA P*V 2.40x, KDA 1.75x, Sinkhorn 2.80x): in those the strided index was the REDUCTION axis, so each line was touched once per output and never revisited. Same AST shape, opposite locality.

PS6011 CONSEQUENCE. PS6011 flags all three of these sites (11 in linalg, 81 tree-wide) and is CORRECT to - it is documented advisory, and its own text says candidates need an A/B. But this is the first recorded case of PS6011 candidates measuring null, and the distinguishing property is stated above: whether the strided index is revisited across the outer loop. That is a genuine refinement, and it is NOT expressible in the current detector - deciding it requires knowing that the same addresses recur, which means reasoning about the outer loop trip count against the index expression. Deliberately NOT adding a suppression: it would need to be sound to avoid hiding the 2.40x cases, and an unsound one is worse than an advisory false positive. The refinement belongs in PATTERNS.md as triage guidance.

WHAT SHIPPED INSTEAD. The scratch hoist found while reading these functions: CholSolve and Lstsq allocated their forward buffer per column, so the Inverse-shaped call (cols == n) paid n allocations. Hoisted: CholSolve n=512 1032 to 521 allocs and 8.40MB to 6.31MB, Lstsq 1546 to 1035 and 10.52MB to 8.43MB, time neutral. Plus the benchmarks - the pre-existing BenchmarkLUSolve topped out at n=128 where the output is 131KB and fits L2 either way, so it could not have detected a traversal effect in the first place.

## R-01KYQTN083E71SB8M9QKZSZYRD Blocking beats interchange 1.62x on the same loops: two independent rewrites, identical bits, and how the winner was picked
kind: research
state: draft
created: 2026-07-29

MEASURED on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, min of 3 per arm, 3 interleaved alternations in one session.

SITUATION. The two RetentionChunkwise output accumulations (nn/retention.go) were rewritten twice independently against the same defect - a column walk at stride d_v, once per output channel (PS6011). Main PR 580 chose INTERCHANGE: accumulate the cross term i-outer into a d_v-length crossbuf, scale it into orow in a second pass, then add the inner-chunk V term lm-outer in a third pass. This branch chose 4-WAY BLOCKING over the output channel: four register accumulators carrying BOTH terms, one pass, crossbuf eliminated. Main measured 1.12x against the original, this branch 1.54x/1.70x against the original - but those are two different baselines, so neither number decides it.

HEAD TO HEAD, blocked against interchanged (not against the original):
  RetentionChunkwise  2.844ms vs 4.613ms  = 1.62x
  RetentionRecurrent  9.494ms vs 15.930ms = 1.68x
  allocs 9 vs 10 (crossbuf)
Consistent across all three alternations with no overlap between arms. Not noise.

WHY BLOCKING WINS HERE, and this is the transferable part. Interchange fixes the ACCESS PATTERN but pays three passes over the output row and keeps the intermediate in memory. Blocking fixes the access pattern AND keeps the intermediate in registers AND fuses the two terms into one pass, so the output row is written once instead of read-modify-written twice. When the strided buffer is already cache-resident, the memory-order win that interchange delivers is the smaller half of what is available - exactly what PERF-ACCUM-RESIDENCY-001 says. PATTERNS.md already states that the choice between the two fixes is a measurement rather than a rule; this is the first head-to-head number for it.

THE CROSS-CHECK THAT MATTERED. Both rewrites carried a bit-identity claim in their comments. Two independent claims of bit-identity with the same original are a claim of bit-identity with EACH OTHER, and that is testable where the individual claims are just prose. Digesting the chunkwise output (bitwise sum plus xor of all elements) under both arms gave 40b2bae2cf78b812 / bf5a8e6dc0e400bb identically. Both claims held. Had they differed, one comment was wrong and the tests would not have caught it - the existing Retention tests are tolerance-based. The digest ships pinned as TestRetentionChunkwiseArmDigest, so the next rewrite of these loops inherits the guard.

NO NEW PERFSCAN RULE. The pattern is already PS6011; what is new is fix SELECTION between two valid rewrites of the same finding, which is a triage question and not a detectable source shape - the scanner sees the defect identically in both cases and by definition cannot see a rewrite that has not been written. Landing in PATTERNS.md as the measured data point behind the existing interchange-versus-blocking guidance.

RELATED: R-01KYQT329EEAD (the PS6011 null case - when neither fix pays because the strided lines stay resident). Together the two records bracket the rule: first ask whether the stride costs anything at all, then if it does, prefer blocking to interchange when the body accumulates.

## R-01KYQVST75E6CTJJJ672ERFV9E rl per-step dispatch waste: PPO rollout 1.59x, DQN learn 1.35x, and two new perfscan rules - one of which replaced the refinement the task specified
kind: research
state: draft
created: 2026-07-29

CONFIRMS T-01KYJQ799JEFM on both defects. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, min of 3 per arm, 3 interleaved alternations in one session.

MEASURED, interleaved against the pre-change source:
  PPORollout        1.146ms -> 0.722ms  1.59x, allocs 29239 -> 15471
  PPOUpdate         3.857ms -> 3.228ms  1.19x, allocs 33177 -> 19409
  DQNLearn batch32  68.9us  -> 52.9us   1.30x, allocs 328 -> 281
  DQNLearn batch128 268.7us -> 199.8us  1.35x, 903KB -> 692KB per step
The task predicted 30-45 percent off the rollout (measured 37), 15-25 off Update (16), and 12-20 off learn (23-26, better than predicted). One benchmark round produced a 3x outlier on the baseline arm from machine interference; min-of-3 absorbed it, which is what the min-of-N discipline is for.

BOTH FIXES ARE BIT-IDENTICAL, and the risk was never the arithmetic. For PPO it is the RNG STREAM: the actor forward, softmax and sampleAction stay in place and in order, so actions, rewards, dones and logpOld are unchanged. TestRolloutTrajectoryParity pins that digest and is mutation-checked - inserting one extra rng draw turns it red. It also pins the critic values digest, because batching them from 256 batch-1 forwards into one batch-270 forward is only sound if the m=1 and m=N kernels agree at tolerance 0. They do on this host (both reach the scalar band kernel) and that is now asserted rather than assumed, with the failure message naming the kernel disagreement so another host reads the right cause first.

RULE (i) BUILT AS SPECIFIED: PS6014 redundant-pure-recompute. Two syntactically identical calls to a config-declared pure function in one block with nothing between that could change the answer. The leading argument is ignored when comparing, which is essential - the motivating pair differed in exactly that argument (fresh Context versus the tape context) and was otherwise identical.

RULE (ii) REPLACED, and the reason is the finding. The task specified refining PS1003 to distinguish hoistable from non-hoistable single-element-batch calls, on the premise that PS1003 fires on both the actor and the critic and only one is fixable. It does not. PS1003 reports ONCE PER LOOP, not per call site, so it never flagged the critic at all - the actor appears first in the same loop and takes the single report. Refining PS1003 would therefore have suppressed a true-but-unfixable warning rather than surfacing the case that paid 1.59x. Built PS6015 instead, reporting per call site and only for the hoistable case. The two rules also give genuinely DIFFERENT advice, which is the deeper reason not to merge them: PS1003 says keep N calls and drop the wrapper allocation (right when the loop reads the result - the actor feeds the sampled action which feeds the environment and cannot move), PS6015 says remove N-1 calls (right when it does not). Both firing on one loop is correct behavior.

METHOD FINDING, the most transferable item here. Replaying a new detector against the real pre-fix source is necessary but NOT SUFFICIENT. PS6014 passed its replay while carrying a real gap: names reachable only through len or cap were counted as possible writes, so New(F64, Shape{len(states), k}) between the two calls suppressed the finding. The real source passed anyway because it happened to size its tensor from a DIFFERENT slice than it fed the forward - an incidental naming difference. The synthetic positive test is what exposed it. Replay proves a rule finds the case it was built from; it does not prove the rule finds the SHAPE, and the gap between those two is exactly one variable name wide.

BOTH RULES ARE CONFIG-GATED on a new pureComputeFuncs vocabulary and registered in the starved-vocabulary warning, so an unconfigured run says the check cannot report rather than printing a zero that reads as clean. Purity is not derivable from syntax and matters more for PS6015 than PS6014: hoisting a call that consumed RNG out of a loop moves draws out of the stream and changes every later iteration.

## R-01KYQW8M07E5VVBA6MKRKXW0N4 PS6014 matched different receivers: a vocabulary of one hid it, and a vacuous suppression test nearly shipped with the fix
kind: research
state: draft
created: 2026-07-29

A CORRECTION to the rule shipped in R-01KYQVST75E6C, found the iteration after shipping it.

THE BUG. PS6014 built its comparison key from calleeName, which collapses a qualified call to its last segment. So b.Wq.Forward(ctx, xn), b.Wk.Forward(ctx, xn) and b.Wv.Forward(ctx, xn) - the three projections at the top of every attention block, and among the most common three consecutive lines in this repository - all keyed as Forward plus the shared argument, and reported as recomputes of each other. The fix is one line: key on the full callee expression. The vocabulary lookup still uses the trailing name, since that is how a method is spelled in config; only the identity comparison needs the receiver.

WHY IT SURVIVED ITS OWN VALIDATION, which is the point. The rule had six tests including five suppression paths, and a replay against the real pre-fix source that reported the exact motivating pair. None could see this, because the configured vocabulary held exactly one name and that name is a PACKAGE-LEVEL FUNCTION taking the network as an explicit argument. No receiver was ever compared, so the receiver-collapsing key was never exercised. A vocabulary of one hides receiver-shaped bugs by construction.

THE METHOD THAT FOUND IT, worth reusing: run the rule against a deliberately over-broad vocabulary and curate every hit. Fourteen plausible names produced dozens of hits across nlp/jamba, nlp/mamba_decode and nlp/quant_cohere - every single one this false positive. After the fix, the same over-broad list produces ZERO, which is also the honest answer to whether other real instances exist in the tree: none do.

CURATION IS ALSO HOW THE VOCABULARY GETS DECIDED. Forward was evaluated and REJECTED rather than added: a Sequential.Forward containing a Dropout consumes RNG in training mode, and a quantized decode Forward mutates its KV cache. Either makes a second identical call load-bearing, so declaring it pure would license deleting a call that must run. The list is a purity ASSERTION, not an inventory of expensive functions - which is exactly why it is config and not a heuristic, and why widening it permanently would be wrong even though it is the right diagnostic move temporarily.

A SECOND DEFECT, caught only by pairing. The regression test for the receiver false positive was first written with a capitalized method name - not in the configured vocabulary - so nothing in it was a pure call at all and the expected zero was reached without exercising the suppression. It passed for the wrong reason. The companion FLOOR test, asserting the rule still fires when the receiver IS the same, failed and exposed it. A zero-expecting test cannot distinguish correct suppression from total non-matching. Reverting the key fix now turns the suppression test red with 2 findings, so the guard is real in both directions.

LANDED AS: PROC-SCANRULE-VOCAB-WIDEN-001 (widen once, curate, restore) and PROC-SUPPRESSION-FLOOR-001 (every suppression test needs a firing floor). Together with PROC-SCANRULE-REPLAY-001 from the previous iteration these bracket scan-rule validation: replay proves the rule finds its instance, a synthetic positive proves it finds the shape, a widened vocabulary proves it does not find non-instances, and a floor proves the suppressions are not vacuous. Each of the four caught something the other three missed.

## R-01KYQWY75AF2G9H56MF9Y3RSV4 nlp attrs-box hoist, batch 1 of the 24-path sweep: -2.9% allocs measured, PS6016 built, and one stale task closed
kind: research
state: draft
created: 2026-07-29

PARTIALLY CONSUMES T-01KYJQZF10F20. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, interleaved, two rounds, deterministic in both.

STALE TASK CLOSED FIRST. T-01KYJQZFEHEFR (route QuantLlama.embedOne through embedRow) was already done - landed by PR 545 (typed row-copy for embedOne, 5.6x, plus KV-evict GatherRows 9.9x). Verified by reading the site rather than assuming the task was current. This is the second stale task found by checking before implementing; the queue predates several merged PRs.

MEASURED, batch of six decode paths (quant_llama, mixtral, olmo2, quant_mixtral, starcoder2, qwen2moe):
  QuantLlamaGenerate500   294573 -> 286083 allocs  -2.9%, -391KB per op
  MixtralPromptStepwise   131227 -> 128589 allocs  -2.0%, -121KB per op
Wall time is a wash and was expected to be - these are small short-lived objects, so the change is garbage pressure and is reported as such rather than dressed up as throughput. 17 of the package 95 escaping attrs sites removed; 78 remain for later batches. The task predicted 1-3 percent per path, which is what landed.

THE NON-OBVIOUS HALF, which the task called out and which held: hoisting the STRUCT is not the fix. quant_llama_decode.go already hoisted its AttnAttrs as a concrete struct and escape analysis still reported it escaping, because the interface conversion happens at the CALL SITE. The box has to move out. An earlier pass on the float paths made exactly this mistake and left the defect in place while looking fixed.

PERFSCAN: PS6016 loop-invariant-literal-arg. A composite literal built inside a loop, passed straight to a call, every field initializer loop-invariant. 155 candidates tree-wide, so the remaining sweep is now automatically located rather than hand-listed. Soundness rests on the literal being passed DIRECTLY and nowhere else - an appended literal needs its per-iteration identity and hoisting it would make every element alias one value, which is a correctness change wearing an optimization costume. Dedup is per SITE not per type: the q and k RoPE attrs are two distinct literals of one type and a per-type key reported one while hiding the other.

RULE (i) OF TWO IS DELIBERATELY HALF-BUILT, and the honest statement of why is the durable part. The already-hoisted-but-still-boxed form is NOT detectable by this scanner: recognizing it requires knowing the parameter is an interface type, which requires go/types, and perfscan is deliberately go/ast-only per its design. Approximating it would either miss or fire on correct code. The tool that sees both forms is the compiler - go build -gcflags=<pkg>=-m names every escaping literal, and that is how both forms were actually found here. PATTERNS.md names the exact invocation rather than pretending the parser covers it. This is the first case in the rule set where the right answer was to document a different tool instead of adding a weaker check.

RULE (ii) NOT BUILT this round: unpooled variadic sibling (a variadic helper called at fixed arity where a fixed-arity sibling exists). Detectable from the signature set alone as the task says, and still open. The exec1-to-exec1a/exec3 switch was applied by hand in this batch; 81 unpooled OpRoPE and 59 unpooled OpMHA call sites remain package-wide, so a rule for it would pay.

## R-01KYQY0G81E9TAJH14B744VCWN nlp attrs hoist batch 2: two false readings before the real one - identical allocs meant no coverage, and 1.73x slower meant one sample
kind: research
state: draft
created: 2026-07-29

CONTINUES T-01KYJQZF10F20 and R-01KYQWY75AF2G. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, interleaved, three rounds.

SHIPPED: 23 more call sites across 18 files. Escaping attrs literals in the package go 78 to 40, and 95 to 40 across the whole sweep.

MEASURED, on a benchmark written for this:
  QuantCohereDecodeStep   1436 -> 1392 allocs  -3.1%, -1530 B/op
Identical numbers in all three rounds. Wall time a wash, as in batch 1 - garbage pressure, not throughput, and reported as such.

FALSE READING ONE, and the more useful of the two. The first A/B used the benchmarks that already existed and reported allocation counts BIT-IDENTICAL across both arms of three benchmarks. That reads as the change does nothing. It actually meant the benchmark runs different code: BenchmarkCohereDecode builds the FLOAT Cohere while every batch-2 file is a quantized path, and BenchmarkQuantLlamaGenerate500 covers a file already fixed in batch 1. THE DIAGNOSTIC IS THE EXACTNESS: a genuine null still jitters by a few allocations run to run, because map iteration and GC timing move it. An exactly repeated count across arms is the signature of code that never executed. Writing BenchmarkQuantCohereDecodeStep produced the real number immediately. Cast as PROC-BENCH-COVERAGE-NULL-001.

FALSE READING TWO. MixtralPromptPrefill appeared 1.73x SLOWER at benchtime=1x - 137ms against 79ms, and the two rounds agreed closely enough to look like a real regression rather than noise. At benchtime=10x: 77.98ms new against 78.91ms old, no regression. One iteration of an 80ms benchmark is one sample, and two such samples per arm can agree by coincidence. Cast as PROC-BENCH-ONE-SAMPLE-001. Worth noting these two failures point opposite ways - one manufactured a null, one manufactured an effect - and both came from trusting a number without asking what produced it.

A THIRD, SMALLER MISS: the mechanical transformer matched only `for l, b := range m.Blocks` and silently skipped `for _, b := range m.Blocks`, which is how the prefill paths spell the same loop. That was a third of the remaining sites, and it surfaced only because escape analysis still reported literals in files the transformer claimed to have fixed. Cross-checking the transformer against the compiler rather than against its own report is what caught it.

WHAT REMAINS. 40 escaping sites, mostly the NON-LOOP form: attrs built inside a per-layer helper function rather than a loop, so hoisting means lifting them into the caller. PS6016 correctly declines these (there is no loop in the function it can see), and they are the same class as the go/types-requiring variant recorded in PROC-SCANRULE-WRONG-TOOL-001 - escape analysis sees them, a parser does not. Also still open: the unpooled-variadic-sibling rule from the original task, with 81 unpooled OpRoPE and 59 unpooled OpMHA call sites package-wide.

## R-01KYQYKZX6FSTSG5HMRMZ3K45Y PS6017 unpooled-variadic-sibling: 422 sites, and two ways a type-comparison helper can break a rule in opposite directions
kind: research
state: draft
created: 2026-07-29

CLOSES the second perfscan rule required by T-01KYJQZF10F20, left open in R-01KYQWY75AF2G.

WHAT IT FINDS. A variadic helper called inside a loop at an arity a non-variadic sibling already covers. The variadic form allocates a slice per call; the sibling takes the same arguments as named parameters and exists to avoid that. 422 candidates tree-wide across four families: exec1 against exec1a/exec2/exec3 in nlp (89 + 292 + 19), and rdropExec/hcExec against execPool1/execPool2 in nn (22), the latter a family nobody had connected. The task estimated 140 from RoPE and MHA call sites alone; exec1 carries many more ops than those two.

NO CONFIG NEEDED, unlike PS6014 and PS6015. The sibling relation is derivable from signatures: identical leading parameter types followed by exactly n parameters of the variadic element type, so the call transfers argument for argument. Built package-wide in a pre-pass, since the variadic form and its siblings are declared in one file and called from twenty - the same reason intMapReg exists.

CONDITION ONE, which removed the only wrong pairing in the tree: at least one FIXED leading parameter. With none, same-trailing-types is far too weak a relation - concat1D(parts ...*tensor.Tensor) matched every two-tensor function in its package. The shared prefix is what constitutes a family; for exec1 it is (ctx, op, attrs), which names the operation all the siblings perform. Found by reading the one outlier in a 401-hit report rather than the three plausible-looking family totals above it.

CONDITION TWO, and the transferable part. Parameter types must be rendered with go/printer. exprText has no StarExpr case and returns empty for every pointer, which breaks the comparison in OPPOSITE directions depending on how the empty is handled:
  - as a placeholder, all pointer types collapse to one value, *backend.Context compares equal to *tensor.Tensor, and unrelated functions pair up (this produced the concat1D hit);
  - as unrenderable-and-skipped, every candidate with a pointer parameter drops out and the rule reports ZERO across the entire tree.
The second is the more dangerous failure because a silent check reads as a clean codebase, and it is exactly the false assurance the starved-vocabulary warning exists to prevent for the config-driven checks.

THE TESTING CONSEQUENCE, verified rather than reasoned: a suppression test written against the pointer case stays GREEN under both failure modes, because it asserts zero and a silenced rule produces zero. Swapping typeText back for exprText leaves every suppression test passing and turns only the POSITIVE test red. So the positive test is the guard for a helper that can return empty, and PROC-SUPPRESSION-FLOOR-001 needs this corollary: the floor is not merely nice to have alongside suppressions, it is the ONLY test that can detect a whole-rule silencing. Cast as PROC-SCANRULE-SILENT-REGRESSION-001.

NOT APPLIED. The 422 candidates are located, not switched. Applying them is a separate change needing its own measurement, and the earlier batches showed why: the alloc delta only appears on a benchmark that executes the changed path, and identical counts across arms mean no coverage rather than no effect (PROC-BENCH-COVERAGE-NULL-001).

## R-01KYQZ94HQE7CSK3DT6CBAHKE8 First application of PS6017: 238 sites swapped, -1.8% allocs where coverage exists, and benchmark coverage is now the binding constraint
kind: research
state: draft
created: 2026-07-29

CONSUMES PS6017 from R-01KYQYKZX6FST. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, interleaved, three rounds.

APPLIED: 238 two-argument exec1 call sites across 59 nlp files routed to the pooled exec2 sibling. exec1 is variadic so every call allocates a slice for its inputs; exec2 takes both tensors as named parameters and pools the slice it builds, delegating to exec1 when ctx.Recorder != nil. Bit-identical by construction, and the package equivalence tests (decode-vs-Forward per architecture) stay green.

MEASURED where coverage exists, deterministic in all three rounds:
  QuantCohereDecodeStep   1393 -> 1368 allocs  -1.8%, -273 B/op
  QuantCohereForward       300 ->  295 allocs  -1.7%,  -20 B/op
PS6017 candidates in nlp fall from about 400 to 230, the remainder being the arity-1 and arity-3 forms plus call shapes the mechanical rewrite did not match.

TWO NULLS THAT ARE CORRECTLY NULL, and telling them apart from the coverage trap is the point. QuantLlamaGenerate500 and MixtralPromptStepwise show no change, because neither executes a changed call site: quant_llama.go got two swaps but both are in Forward, and Generate steps the prompt through DecodeStep. The DISTINGUISHING SIGNAL is jitter. These counts vary by a handful of allocations between runs (283397 / 283413 / 283397 against 283399 / 283405 / 283403) rather than repeating exactly. An EXACTLY repeated count across arms means the code never ran (PROC-BENCH-COVERAGE-NULL-001); an overlapping jittery range means the effect is below noise on that benchmark. Both look like zero in a summary table and they mean different things.

THE BINDING CONSTRAINT IS BENCHMARK COVERAGE, not the edits. Forward and DecodeStep are separate layer loops with separate call sites, so a change to one is invisible to a benchmark of the other. Of 59 changed files, exactly one has benchmarks for both paths - and that is only because BenchmarkQuantCohereForward was written this round to pair with the DecodeStep benchmark written last round. The remaining 230 candidates are mostly in paths no benchmark executes, so under the standing constraint that only what is verifiable on this system may ship, the sweep is now gated on writing per-architecture prefill and decode benchmarks rather than on applying rewrites. That is a larger and less interesting body of work than the optimization itself, and it is where the next iterations of this line should go.

NO TIME CLAIM. MixtralPromptStepwise is bimodal across a 3x range WITHIN a single arm on this host (166ms to 519ms), so it cannot resolve a percent-level effect in either direction. Reporting a ratio from it would be noise dressed as a measurement, per PROC-BENCH-ONE-SAMPLE-001 generalized: the guard is not just iteration count but whether the within-arm spread is smaller than the claimed effect.

## R-01KYQZTPMPETEA5P4QNR55VF8B Removing the coverage constraint unlocked the sweep: 12-architecture benchmark matrix, then 135 swaps verified on every one
kind: research
state: draft
created: 2026-07-29

CONSUMES the constraint identified in R-01KYQZ94HQE7C. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, interleaved, three rounds.

THE UNLOCK WAS COVERAGE, NOT REWRITES. The previous iteration could apply PS6017 candidates but could only verify the two paths that happened to have benchmarks. Eleven of twelve quantized architectures had no benchmark on either Forward or DecodeStep, so further batches would have shipped on escape analysis alone. Writing the matrix first turned an unverifiable sweep into a measured one, and it took less effort than the optimization work it gated.

BUILT: 24 benchmarks, prefill and decode for all twelve quantized transformer architectures, table-driven over the existing GGUF test fixtures. Three design points worth keeping. (1) SMALL models deliberately - the quantity being guarded is per-layer allocation count, which is geometry-independent, so a production-sized model would multiply runtime without changing the number. (2) NO shared interface: the cache type differs per architecture (CohereCache, FalconCache) and the SSM families take a decode state instead, so each entry supplies its own closures. Twelve small closures beat one wrong abstraction. (3) The quant*GGUFBytes fixtures were widened from *testing.T to testing.TB (plus quantDeepSeekV2Write transitively) so benchmarks reuse them rather than duplicating GGUF construction.

THEN MEASURED, 104 exec1-to-exec1a and 31 exec1-to-exec3 swaps:
  decode  Gemma2 4952->4862, DeepSeekV2 5241->5167, Nemotron 2752->2722, GPTNeoX 3101->3072, MPT 1106->1086, StableLM 2831->2812, Falcon 1151->1141, Gemma 1441->1431, StarCoder2 1651->1641
  prefill Gemma2 1071->1053, DeepSeekV2 1119->1103, Nemotron 564->556, GPTNeoX 635->628, StableLM 582->576, and every other architecture down 2 to 4
Every architecture improves or holds. -292 allocations across the decode matrix, -78 across prefill, deterministic to within two allocations over three rounds.

THREE GENUINE ZEROES: Cohere, Mixtral and OLMo2 decode hold exactly, because earlier batches already consumed their arity-1 and arity-3 sites. These are nulls WITH jitter, which under PROC-BENCH-NULL-KINDS-001 is the signature of a real null rather than a benchmark that misses the code - and having the matrix in place is what made that distinction checkable instead of a judgment call.

CUMULATIVE across the sweep: 373 exec1 call sites moved to pooled siblings, 55 attrs boxes hoisted out of layer loops, PS6017 candidates in nlp down from about 400 to 176. The remainder are call shapes the mechanical rewrite does not match - indexed and call-valued arguments - which need per-site judgment rather than a regex.

METHOD NOTE. Two iterations were spent measuring things that could not be measured before recognizing that the fix was to build the instrument. The signal was available earlier: PROC-BENCH-COVERAGE-NULL-001 was cast one iteration before this, from exactly the same symptom, and treating it as a warning about a single benchmark rather than a systemic gap cost a round.

## R-01KYR0F8WRFCSSPSJ9ASZS6F3R The benchmark matrix turned out to be a diagnostic: a 4.5x allocation outlier led to Gemma2 decode 1.21x, -27.6% allocs
kind: research
state: draft
created: 2026-07-29

Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, five interleaved rounds at benchtime=300x.

HOW IT WAS FOUND, which is the reusable part. The twelve-architecture benchmark matrix was built as a GUARD - to make the allocation sweep verifiable. Reading it as data instead showed Gemma2 decode at 4862 allocations against MPT 1086 on IDENTICAL geometry (all twelve fixtures are two layers, dim 32, five tokens), a 4.5x spread that geometry cannot explain. That pointed a profiler at one function. An instrument built for verification answered a question nobody had asked, and the normalization that made it readable was checking that the fixtures share geometry - without that the spread looks like model size.

THE DEFECT. Gemma2 cannot use the fused OpMHA every other architecture takes, because the attention-logit soft-cap sits between the scores and the softmax. So cappedDecodeAttention hand-rolls attention per head: three slices, a transpose, two matmuls, a scalar multiply, a soft-cap and a softmax, per head per layer per token. The allocation profile put that single function at 78.65% of a decode step, with ref.sliceKernel 17.58% plus SlicePlan 2.51% plus ref.transposeKernel 5.86% - 26% in pure data movement.

THE FIX, scoped by ADR-01KYQ9PHNPEFC: fuse ONLY the movement. The three slices and the transpose are pure gather, so qh/khT/vh are built straight from storage with the transpose done during the copy. Bit-identical by construction - the same values reach the same kernels in the same order - while matmul, mul, soft-cap and softmax stay on the backend, where fusing them would risk reassociation and FMA contraction for no structural gain. One scratch tensor per role reused across heads, each fully overwritten before its head reads it.

MEASURED: 4862 to 3522 allocations, -27.6%, identical in all five rounds. 192-202us to 158-168us, non-overlapping ranges, so 1.21x is the conservative figure. Prefill unchanged at exactly 1054 both arms, correctly - Forward does not call this function.

A MEASUREMENT TRAP, this time on a change that WAS real. At benchtime=50x the same fix looked like 1.90x: the fused arm ranged 157-247us while the baseline sat tight at 299-305us, and min-of-N reported whichever round caught the fast mode. At 300x both arms are tight and the honest ratio is 1.21x. PROC-BENCH-ONE-SAMPLE-001 was cast from a case where low benchtime manufactured a regression that did not exist; it applies equally where low benchtime inflates a real gain, and the failure mode is symmetric.

BIT-IDENTITY TESTED, NOT ARGUED, and the test design is worth reusing: the two arms are selected by ctx.Recorder, so a plain context takes the fused gather and a taped context keeps the dispatch path. Comparing raw float32 bits between them exercises the arithmetic AND the guard in one pass. Panic-probed to confirm the fused branch is reached, because a parity test whose arms take the same path passes while proving nothing.

NO NEW PERFSCAN RULE: PS4011 already flags this file and this shape - a sequential loop dispatching several backend ops per iteration with no fused path - and its recorded remedy is exactly what was applied. The rule was right and unconsumed.

## R-01KYR0ZEEFFK6RCAGDRKH14WF0 Matrix diagnostic, second target: DeepSeekV2 decode 1.12x / -9.3% allocs, and min-of-N nearly reported a regression
kind: research
state: draft
created: 2026-07-29

Second application of PROC-BENCH-MATRIX-DIAGNOSTIC-001. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, benchtime=400x, six interleaved rounds.

TARGET SELECTION took no code reading: after the Gemma2 fix, DeepSeekV2 decode was the remaining outlier in the twelve-architecture matrix at 5166 allocations against MPT 1086 on identical two-layer dim-32 fixtures. A profile put attnAbsorbed at 58.48% of a decode step, with QMatMul 23.91% inside it (real work) and ref.sliceKernel 10.93% plus SlicePlan 1.56% as pure movement.

THE FIX. Three slice dispatches per head become two copies. One of the three was pure indirection worth naming: qh was sliced out of q only to be re-sliced into qNope and qPe, and both of those are contiguous runs of q own storage, so the intermediate never needed to exist. Movement only per ADR-01KYQ9PHNPEFC; matmul, RoPE and add stay on the backend.

MEASURED: 5166 to 4686 allocations, -9.3%, identical every round. 281-292us to 250-281us, 1.12x by min-of-six, ranges all but disjoint.

THE MEASUREMENT TRAP, third instance this session and the most dangerous form. At benchtime=300x one BASELINE round came in at 257us against its own cluster of 305-312us. Min-of-N over those four rounds picks 257 for the baseline and 297 for the fused arm, reporting the change as a 15 PERCENT REGRESSION - the exact opposite of the truth. Raising to 400x and taking six rounds separated the arms cleanly. Tally of the three instances: low benchtime manufactured a regression that did not exist (Mixtral prefill, 1.73x phantom), inflated a real gain (Gemma2, 1.90x reported where 1.21x was true), and here nearly inverted the sign of a real gain. MIN-OF-N IS NOT A DEFENSE against a bimodal distribution - it actively amplifies the problem, because it selects on exactly the tail that the bimodality produces. The defense is raising benchtime until the within-arm spread is smaller than the claimed effect, and CHECKING that before computing any ratio.

BOTH FUSIONS SHARE A TEST DESIGN worth reusing: the two arms are selected by ctx.Recorder, so a plain context takes the fused path and a taped one keeps the dispatch path. One raw-bit comparison therefore tests the arithmetic AND the tape guard, and a panic probe confirms the fused branch is reached - a parity test whose arms take the same path passes while proving nothing.

DELIBERATELY NOT DONE: the non-absorbed per-head loop in the same file, which slices kv as well as q. It needs its own parity test and its own measurement, and the benchmark exercises the absorbed path, so shipping it here would have been unverified.

MATRIX STATE after two fixes: Gemma2 decode 4862 to 3522, DeepSeekV2 5166 to 4686. The remaining spread is GPTNeoX 3072 and StableLM 2812 against MPT 1086, which is the next place to look.

## R-01KYR1VZPWE2TR4REXMP8BDN1S partialRoPE fusion is the sessions largest win: 1.25-1.33x and -38 to -43% allocs across three architectures, and it generalized into PS6018
kind: research
state: draft
created: 2026-07-29

Third and largest application of PROC-BENCH-MATRIX-DIAGNOSTIC-001. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, benchtime=300x, four interleaved rounds.

TARGET. GPTNeoX decode was the next matrix outlier at 3072 allocations. A profile put partialRoPE at 51.39% of the step - and unlike the two previous targets this is a SHARED HELPER with 31 call sites, so the fix lifts every architecture that uses partial RoPE rather than one model.

THE DEFECT. Eight backend dispatches - reshape, two slices, reshape, RoPE, reshape, concat, reshape - of which exactly ONE does arithmetic. The other seven are pure layout algebra.

THE FIX. Gather each head rotated prefix into one [seq, heads*rotaryDim] buffer, call the same RoPE, scatter the result back beside the untouched tail. One dispatch instead of eight. The layout equivalence is the entire proof and is spelled out in the code rather than asserted: the dispatch path rotWide has row s equal to the concatenation over h of x[s, h*hd : h*hd+rotaryDim], which is what the gather builds; the RoPE is the same op with the same attrs on the same shape; the scatter rebuilds what concat-plus-reshape produced.

MEASURED, allocation counts identical every round, timing ranges non-overlapping, within-arm spreads about 5% so the PERF-MINOFN-NOT-A-DEFENSE-001 spread check passes:
  GPTNeoX   3071 -> 1891 allocs  -38.4%   181-191us -> 145-153us  1.25x
  StableLM  2811 -> 1631 allocs  -42.0%   122-126us ->  95- 96us  1.28x
  Nemotron  2721 -> 1541 allocs  -43.4%   120-121us ->  90- 92us  1.33x
Exactly -1180 allocations on all three - same helper, same call count, which is itself a consistency check. The other nine architectures are unchanged within jitter and do not use partial RoPE.

PARITY SWEPT OVER SEVEN GEOMETRIES, not one, because the claim is index arithmetic: a single (seq, heads, rotaryDim) triple can agree by coincidence when an offset is wrong, and rotaryDim == hd takes a different early-return branch. The sweeps first version included an ODD rotaryDim and failed on the backends own even-head-dim validation - a bad test rather than a finding, and worth recording because a sweep that fails for an invalid-input reason looks exactly like a sweep that found a bug.

GENERALIZED INTO PS6018 layout-op-cluster-unfused, three or more pure movement dispatches with no fused raw-storage path. The key property is that MOVEMENT CANNOT CHANGE A VALUE, so the fix is bit-identical by construction - index arithmetic, no numerical judgment - which is what makes the class flaggable on sight where a bare dispatch-count report would not be. PS4011 does not subsume it: that rule needs a sequential loop and partialRoPE is straight-line, so the largest win of the three was invisible to it. 27 candidates tree-wide; the one surviving hit among the three fixed files is attnReconstructed, the DeepSeekV2 loop deliberately left unfused, which is the rule reporting exactly the known gap.

CUMULATIVE, matrix decode allocations: Gemma2 4862 to 3522, DeepSeekV2 5166 to 4686, GPTNeoX 3072 to 1891, StableLM 2812 to 1631, Nemotron 2721 to 1541. The spread that started the investigation - 4.5x between the worst architecture and MPT - is now about 3.2x, and the remaining head is Gemma2 and DeepSeekV2, both of which still carry unfused per-head arithmetic that would need a numerical argument rather than an index one.

## R-01KYR2G61XE669Z658R4GPG6EG Swin per-head bias cache: -21.5% allocs, an exact-snapshot invalidation key, and PS6018 candidates that were all cold
kind: research
state: draft
created: 2026-07-29

Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, four interleaved rounds at benchtime=20x.

A NEGATIVE FIRST, because it is the more useful half. PS6018 reported 27 candidates tree-wide. 20 are in nn, the user parallel lane, so out of scope. Of the 5 in vision, ALL FIVE are cold: they sit in top-level Forward and forwardBatched layout, not the per-block path, and the profiles put Swin at windowedAttention 72% and MLPMixer at Linear.Forward 40%. A rule firing is not a target - PROC-TASK-HOTNESS-001 exists for exactly this, and checking cost one profile run per file. The rule is still correct; its vision hits simply are not worth a diff.

WHAT THE PROFILE FOUND INSTEAD. swinRelBias.headBias is a pure function of (Table, oneHot, head) - nothing about the input image enters it - and ran once per head per block PER IMAGE, at 21.6% of a per-image forward allocations.

MEASURED: perimage 19410 to 15229 allocations, -21.5%, deterministic across four rounds, time 38.12-38.79ms to 36.34-36.75ms, non-overlapping, 1.05x. Batched 10142 to 9620, -5.1%, time NOT resolvable because the ranges overlap and no claim is made. The -21.5% lands on the 21.6% the profile predicted. Time moves little because allocation was never the time bottleneck here - matmulKernel is 39.9% - and batching already amortizes headBias across the batch, which is why that arm gains less.

THE INVALIDATION KEY IS THE DESIGN DECISION. Table is the TRAINABLE parameter, so this is a cache over mutable state and the failure mode is silent: a stale bias changes only the numbers, never the shapes or the control flow. Rejected a generation counter (nothing increments it without touching the optimizer) and a checksum (a collision is a wrong answer, and for a cache that is unacceptable when the exact check is affordable). Chose an EXACT element-wise comparison against a stored copy of Table: (2M-1)^2*heads floats compared, against a matmul over M^4*(2M-1)^2 multiply-adds avoided - orders of magnitude cheaper, and a proof rather than a guess.

TWO SAFETY CONDITIONS, both checked rather than assumed. Only inference contexts touch the cache: under a tape the bias must be a real graph node for the gradient to reach Table, so a taped call recomputes and never populates. And a cached tensor is only safe to share if nothing mutates it - swinFuseScoreTerms was read to confirm it writes into the scores and merely reads the bias.

MUTATION-VERIFIED: replacing the exact comparison with a cache-once key makes the invalidation test fail with exactly the stale-value message it was written for. Two further tests pin the cache-hit values and that a taped context is never handed the cached tensor.

NO PERFSCAN RULE, and the reason is structural rather than effort. Recognizing this needs to know the method is pure with respect to its RECEIVER and that callers repeat across images - call-graph and lifetime reasoning, not syntax. PS6014 is the within-a-block relative and cannot reach across calls. Per PROC-SCANRULE-WRONG-TOOL-001 the honest answer is to name the tool that does find it: an allocation profile, which is how it surfaced, and which is now the second time this session that profiling beat scanning on a real target.

## R-01KYR2YAEBE4ZA09FXT7TFPEQB GMM full-cov PredictProba 3.1-5.8x: the blocker was a receiver scratch buffer, and the unparallelizable shape was also the unmeasured one
kind: research
state: draft
created: 2026-07-29

CONSUMES T-01KYPCP21VFAZ, deferred during an earlier rebase because main had added a component jam this had to be re-applied on top of. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, benchtime=200x, four interleaved rounds.

MEASURED, ranges disjoint, within-arm spreads 2-12% against a 3-6x effect:
  512x8  d16   412-427us  ->  131-134us   3.14x
  512x8  d32  1328-1371us ->  291-299us   4.57x
  2048x8 d32  5355-5981us ->  916-1020us  5.84x
Speedup grows with row count, as row-parallel scaling should.

THE BLOCKER WAS ONE FIELD. PredictProba parallel row scan was gated to GMMDiag, and the gate own comment named the cause: the full-cov density kernel read its four triangular-solve buffers off the RECEIVER, so concurrent calls would have raced. Rows were always independent for both covariance shapes - the kernels only read per-component params. Passing the buffers in as a parameter removed the obstacle and the covariance condition dropped out of the gate. This is PS6006 (receiver-scratch-buffer) not as a micro-optimization but as a STRUCTURAL BLOCKER: a per-call temporary on shared state does not merely contend, it forecloses parallelism entirely, and the cost shows up as a gate somewhere else in the file rather than as slow code at the site.

PS6006 TRACKED THE FIX, which is worth noting as a rule-quality signal: it flagged both receiver buffers before the change and now flags only m.yScratch, the single-buffer one still used by the scalar logGaussian. Freeing that one would open the E-step by the same argument, and it needs its own measurement.

THE COMPOUNDING GAP. Full covariance had NO PredictProba benchmark at all. So the one covariance shape that could not be parallelized was also the one nothing measured - and full-cov is where the work is, since the solve is O(d^2) per component against the diagonal form O(d). A 5.8x sat unnoticed because the measurement gap and the optimization gap had the same cause: whoever gated the path to GMMDiag also only benchmarked GMMDiag. Worth generalizing: when a code path is excluded from an optimization for a stated reason, check whether it is also excluded from measurement, because the same author-attention pattern produces both.

BIT-IDENTITY TESTED THROUGH THE REAL GATE rather than a flag: the paths are selected by a work threshold, so the test compares a batch above it against the same rows fed one at a time below it. Race detector clean.

METHOD NOTE. This target was NOT found by profiling or by a scan rule sweep - both had been applied to the packages I own and were largely harvested. It came from re-reading my own deferred task list. Two of the last three iterations opened by profiling a benchmark-matrix outlier; this one opened by asking what I had explicitly postponed and why the reason no longer held.

## R-01KYR3CE7BFQ4SDY12MFJJ0WJ3 A PS6006 sweep looking for the next optimization found a race I had shipped one commit earlier
kind: research
state: draft
created: 2026-07-29

CORRECTS R-01KYR2YAEBE4Z. The GMM full-covariance parallelization measured 3.1x-5.8x and was correct for the cases it tested; it was NOT correct in general.

THE BUG. logGaussianFullBatch is unroll-and-jammed by four over the component index and finishes the remainder by calling the scalar logGaussian. Parallelizing PredictProba required moving the batch kernel four triangular-solve buffers off the receiver - which was done - but the TAIL call still read its own forward-substitution buffer off the receiver. So the parallel row scan raced for any component count not a multiple of four.

WHY THE VALIDATION MISSED IT, and this is the transferable part. Every benchmark and every test in that change used k=8. The k%4 tail therefore never executed, and:
  - the race detector cannot flag a line that does not run;
  - the parity test compared parallel against serial on a path where the tail was empty, so it agreed for the wrong reason;
  - go vet and the build see nothing, since the receiver field is legitimately typed.
The defect was ONE MODULO away from the tested case. An unroll-and-jammed kernel has two code paths, the wide one and the scalar remainder, and a fix applied to the wide one does not reach the tail - but a test at a convenient trip count never notices. Cast as PROC-UNROLL-TAIL-COVERAGE-001. A k=5,6,7 test now reports four DATA RACEs against the old tail, and the fix passes with the speedups unchanged.

HOW IT WAS FOUND. Not by review and not by CI - by acting on PERF-RECEIVER-SCRATCH-BLOCKS-PARALLEL-001, cast the iteration before, which prescribes sweeping PS6006 tree-wide and checking each receiver-scratch finding for a parallelism gate that names it. The sweep listed classic/gmm.go m.yScratch, and tracing what its serialization blocked led straight into the tail. THE SWEEP WAS LOOKING FOR THE NEXT OPTIMIZATION AND FOUND A REGRESSION, which is the better outcome and an argument for running these rule-driven sweeps even when the immediate goal is new work rather than verification.

SCOPE OF THE SWEEP, recorded so it is not repeated: PS6006 reports ten receiver-scratch findings tree-wide. Four are in nn and tree.go, the users parallel lanes, and out of scope. Of the remaining four in classic - gmm.go m.yScratch, dbscan.go m.core, gbm.go b.goLeft, spatialindex.go bt.splitKey - only gmm.go carried a comment naming its buffer as forcing serial execution. The other three are per-call temporaries with no parallelism gate citing them, so under PERF-RECEIVER-SCRATCH-BLOCKS-PARALLEL-001 they are ordinary PS6006 candidates rather than structural blockers, and none is currently hot.

## R-01KYR3YV0HF26ACHX0H4C0NYBG One receiver field gated three optimizations: PredictProba 3.1-5.8x, Fit 1.27x/2.69x, and PS6006 tracked its own resolution to zero
kind: research
state: draft
created: 2026-07-29

Completes the GMM arc opened in R-01KYR2YAEBE4Z and corrected in R-01KYR3CE7BFQ4. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, three to four interleaved rounds, all ranges disjoint.

CUMULATIVE, from moving two scratch buffers off the receiver:
  PredictProba full-cov  3.14x / 4.57x / 5.84x  (512x8 d16, 512x8 d32, 2048x8 d32)
  Fit diagonal           1.27x  (11.28-11.57ms -> 8.90-9.10ms)
  Fit full covariance    2.69x  (47.57-52.09ms -> 17.67-18.19ms)
Full covariance gains most everywhere: its O(d^2) triangular solve dominates and parallelizes cleanly against the diagonal form O(d).

THE STRUCTURAL POINT, now demonstrated three times over: a per-call temporary on a receiver field does not cost cache contention, it FORECLOSES parallelism, and the cost appears as a serial gate somewhere else in the file. Two fields - yScratch4 and yScratch - between them gated the full-covariance PredictProba row scan and the E-step for both covariance shapes. Neither gate was about the math; rows were always independent. Removing the fields removed three gates.

BIT-IDENTITY, where it had to be earned. The E-step running log-likelihood total is the only cross-sample dependency, so it becomes norms[i] summed AFTER the loop in ascending sample order - a chunked reduction would reassociate. n extra words against O(n*k*d) work.

THE PROOF IS A PANIC-PROBED GOLDEN TEST. The package golden test pins Fit output bit-exactly, but a golden test that only runs the serial branch proves nothing about the concurrent one, so the parallel body was panic-probed to confirm the golden case actually reaches it. This is the third time this session that a probe was needed to establish a passing test was not vacuous, and the pattern is consistent: whenever a change adds a guarded second path, the test that covers it must be proven to enter it.

PS6006 TRACKED THE ENTIRE ARC AND NOW REPORTS ZERO in gmm.go: two findings, then one after the PredictProba work exposed the k%4 tail, now none. A rule measuring its own resolution is the property PS6019 deliberately lacks - that one is a standing maintenance hazard rather than a closable defect, and the contrast is worth keeping in mind when deciding whether a new rule should have a quiet state.

SWEEP RESULTS, recorded so they are not repeated. PS6019 (jam-tail-delegates), built this iteration from the shipped race, finds ONE candidate tree-wide - the GMM kernel itself. PS6012 (inconsistent-fma-pinning) finds ZERO across classic, linalg, vision, nlp and rl, so the other stale-tail-property class is clean in my lanes. The tail-hazard sweep is therefore complete: one known delegating tail, no pinning divergence.

## R-01KYR494K5ESBRNPETPAQ39EDC REJECTED: GBM feature fan-out is not over-parallelized. I misread a CPU profile of a parallel program - condvar wait time is idling, not overhead
kind: research
state: draft
created: 2026-07-29

REJECTS a hypothesis I formed and refuted within one iteration. Measured on darwin/arm64 M2 Pro, go1.26.5, GOMAXPROCS=12, three interleaved rounds. Nothing shipped.

THE HYPOTHESIS. A CPU profile of BenchmarkGBMHist_exact_80k (20 features, 80k rows, 50 estimators) showed 52.98% in runtime.pthread_cond_wait, 16.60% in pthread_cond_signal and 4.18% in usleep - about 74% - against 13.94% in gbmBuilder.bestSplit.func1 and 4.06% in partition.func1, roughly 18% in the actual split search. That reads as textbook over-parallelization, and the structure supported it: parallelFeaturesIdx fans out over FEATURES, is called once per tree NODE, and with d=20 on a 12-worker pool each chunk gets one or two features. The threshold was d*n against 1<<15, a crossover borrowed from backend/cpu, which checks TOTAL work and never asks whether a chunk has enough work to earn its wake-up.

THE FIX I BUILT: gate on per-chunk work instead, requiring d*n >= threshold*workers so each chunk clears the same bar.

MEASURED, and it is decisively WORSE:
  GBMFit             67-71ms   -> 126-129ms   1.88x SLOWER
  GBMHist_exact_80k  652-681ms -> 785-806ms   1.20x SLOWER
  GBMHist_hist_80k   unchanged (the histogram grower has its own separate gate)
Ranges disjoint in both directions. The parallelism was paying at these sizes and the per-chunk gate switched it off where it helped.

THE ACTUAL LESSON, which is a measurement error and not a code fact. IN A CPU PROFILE OF A PARALLEL PROGRAM, TIME IN pthread_cond_wait AND pthread_cond_signal IS NOT NECESSARILY OVERHEAD. A CPU profile sums time across all threads, so a pool of workers idling on a condition variable accumulates large flat percentages while consuming no wall-clock progress at all. The 74% was mostly eleven workers waiting for the twelfth, which is what a correctly-sized pool looks like when sampled this way - it is not evidence of anything. I read a parallel profile with instincts calibrated on serial ones.

WHAT WOULD HAVE ANSWERED IT CHEAPER. The hypothesis was about wall-clock, so the first move should have been a wall-clock experiment, not a structural fix: run the benchmark at GOMAXPROCS=1 against the default and see whether the parallelism helps at all. That is one command and it would have refuted this before any code was written. A CPU profile localizes work; only wall-clock arbitrates a parallelism decision.

NO PERFSCAN RULE, and here the reason is sharper than usual: the pattern I thought I had found - a threshold that checks total rather than per-chunk work - IS AST-detectable, and building a rule for it would have institutionalized a wrong belief. A scan rule asserts that a shape is a defect; this shape is not one, at least not here. The gate on total work is correct for this caller and the borrowed crossover is doing its job.

## R-01KYR4Z7SWFTCRF76GHBNYVDRK The GOMAXPROCS sweep turned a failed hypothesis into a diagnostic: SoftmaxRegression Fit 1.96x, and the reduction axis decides legality
kind: research
state: draft
created: 2026-07-29

Measured on darwin/arm64 M2 Pro, go1.26.5, four interleaved rounds at benchtime=20x.

THE INSTRUMENT CAME FROM THE PREVIOUS FAILURE. R-01KYR494K5ESB rejected an over-parallelization hypothesis and the corrective rule prescribed settling parallelism questions with wall-clock at GOMAXPROCS=1 against the default. Running that as a SWEEP over the classic benchmarks is one command per benchmark and immediately ranks every path by how much parallelism it actually gets:
  ForestFit 7.93x, KNNPredict 5.87x, DBSCANFit 5.67x, GMMFitFull 3.51x,
  GBMHist_exact_80k 2.79x, GBMFit 1.73x, GBMHist_hist_80k 1.53x,
  SoftmaxRegressionFit 0.99x, SVCFit/n4000_rbf 0.97x
The last two are entirely serial. A failed hypothesis produced a better target-finder than the profile that misled it.

THE TARGET. A line-level profile put ONE loop - the Hessian Gram accumulation - at 76% of Fit flat time, 190ms of 250ms.

THE AXIS CHOICE IS THE ENTIRE CHANGE, and it is the transferable idea. grams[q*mm + a*mAug + j] accumulates over exactly ONE axis, the sample index, so its sum stays ascending however the pair index q, the feature row a and the column j are ordered outside it - and distinct a write disjoint ranges within each pair block. So interchanging a outside the sample loop and parallelizing over it is bit-identical. Parallelizing over SAMPLES is not: each worker needs its own partial Gram and combining partials reassociates the i-sum. Same loop, same speedup target, one axis legal and the other not. The question to ask of any reduction loop is which index carries the accumulation - everything else is free to move.

MEASURED: 17.99-19.69ms to 8.91-9.17ms, 1.96x, ranges disjoint. Mechanism confirmed rather than assumed - the benchmark now scales 2.5x from GOMAXPROCS=1 to 12 against 0.99x before.

A CAVEAT WORTH THE EXTRA CODE. The interchange is NOT free when it cannot fan out: sweeping X once per feature row costs locality the sample-outer nest kept, measured 23.8ms against 18.0ms at GOMAXPROCS=1. Keeping the original nest for the serial branch leaves single-core at parity (18.9ms) instead of 30% worse. Under the standing constraint that an optimization must improve things across systems, a change that regresses a single-core host does not qualify - and this would have, silently, since the default-GOMAXPROCS A/B showed only the 1.96x.

BIT-IDENTITY established by capturing digests from the serial implementation BEFORE the interchange and reproducing them exactly after, over three geometries: the benchmark shape, k=4 with six pair blocks, and k=2 with a single block. Pinned as assertions.

STILL OPEN from the sweep: SVCFit at 0.97x, also entirely serial, with an existing task (T-01KYJQ78WJF57) proposing a flattened kernel matrix. The gradient and Hessian-scatter loops in this same Fit remain serial and cannot be chunked bit-identically over samples for the reason above.

## R-01KYR5DFZHECA80K8PJTPSMP7C SVC kernel column 1.27x: the GOMAXPROCS sweep is exhausted for classic, and a size below the threshold looked like a regression
kind: research
state: draft
created: 2026-07-30

CONSUMES the second finding of the GOMAXPROCS sweep in R-01KYR4Z7SWFTC, and closes T-01KYJQ78WJF57 by a different route than it proposed. Measured on darwin/arm64 M2 Pro, go1.26.5, three interleaved rounds at benchtime=30x.

MEASURED:
  SVCFit/n4000_rbf  6.58-6.73ms -> 5.20-5.21ms  1.27x, ranges disjoint
  SVCFit/n1000_rbf  unchanged, below the work threshold
  parallel scaling  0.97x -> 1.30x from GOMAXPROCS=1 to 12
  GOMAXPROCS=1      7.13ms vs 7.17ms, parity within noise

THE CEILING IS AMDAHL AND IS WORTH STATING WITH THE RATIO. SMO cannot be parallelized - it selects a maximal-violating pair and updates, one iteration at a time. Only the two kernel columns each step evaluates can be, and they were 58% of the profile. So 1.27x is close to what 58% parallelizable work allows, not a partial result waiting for more effort. The task T-01KYJQ78WJF57 proposed flattening the kernel matrix and specializing the kernel column per kernel type; parallelizing the existing column loop got the available win without touching the memory layout, and the layout change would now be chasing the remaining 42% sequential share.

BIT-IDENTITY IS PINNED ON THE FITTED MODEL, not on a column, and the reason generalizes: SMO iteration DEPENDS on the values it reads, so a divergence does not stay small - it compounds into a different support vector set. For any solver whose control flow reads the values being changed, the model output is the only meaningful parity target. Digests reproduced exactly at both sizes with identical support vector counts.

A SIZE BELOW THE THRESHOLD LOOKED LIKE A REGRESSION. The n=1000 arm showed 1.39ms in one baseline round against 1.93-2.01ms for the new arm - a 1.4x apparent regression. Both arms run IDENTICAL SERIAL CODE at that size: d=20 puts the work at 20000 against the 32768 threshold. A panic probe confirmed n=1000 never enters the parallel path and n=4000 always does. That is a structural answer in one command, against re-running the benchmark until the noise settles - and it is the third distinct way this session that a benchmark number has misled (phantom regression from one sample, inverted sign from a bimodal min-of-N, and now a spread on a code path the change does not reach).

THE SWEEP IS NOW EXHAUSTED FOR classic. Both 0.97-0.99x paths are fixed. The remaining ranking is ForestFit 7.93x, KNNPredict 5.87x, DBSCANFit 5.67x, GMMFitFull 3.51x, GBMHist_exact 2.79x, GBMFit 1.73x, GBMHist_hist 1.53x - all genuinely parallel, and the two GBM figures were separately confirmed NOT to be over-parallelization (R-01KYR494K5ESB). Applying the same sweep to another package is the cheapest next move; it costs one command per benchmark and needs no code reading.

## R-01KYR5ZS7RF7HTDA18JVHTR9EN linalg solve phase up to 7.87x: the sweep found an entirely serial package, and the blocker was a hoist I had made myself
kind: research
state: draft
created: 2026-07-30

Second application of the GOMAXPROCS sweep, and the largest win of the session. Measured on darwin/arm64 M2 Pro, go1.26.5, three interleaved rounds, all ranges disjoint.

THE SWEEP FOUND A WHOLE PACKAGE. Applied to linalg and rl, it showed linalg at 1.00-1.10x from one core to twelve on EVERY benchmark - LUSolve, Inverse, CholSolve, Lstsq, SVDPCA - while holding the largest absolute numbers in my lanes: LstsqMat/768 at 1.3s, Inverse/768 676ms, CholSolveMat/768 629ms. One command per benchmark located that; no code reading and no profiling.

MEASURED:
  LUSolve_768x768   549-564ms  ->  70- 76ms   7.87x  (pure solve, Factor hoisted out)
  CholSolveMat/768  629-657ms  -> 172-177ms   3.66x
  Inverse/768       676-686ms  -> 197-221ms   3.43x
  LstsqMat/768     1325-1350ms -> 702-722ms   1.89x
  Inverse/64          704us    ->   362us     1.94x
  CholSolveMat/64     545us    ->   280us     1.95x
The spread across these is Amdahl, not effort: LUSolve is highest because its benchmark hoists the factorization out, while the others include a sequential factorization whose share bounds them.

THE BLOCKER WAS A HOIST I MADE MYSELF, which is the finding worth carrying. An earlier commit in this same session moved the forward-substitution buffer out of the column loop to cut cols allocations per call - a real, measured allocation win at the time - and in doing so coupled every column to every other, foreclosing a 7.87x. Per-worker scratch restores independence and keeps the allocation win. This is the third instance of the same structural story after GMM yScratch4 and yScratch: A PER-CALL TEMPORARY SHARED ACROSS ITERATIONS FORECLOSES PARALLELISM, AND THE COST APPEARS NOWHERE NEAR THE BUFFER. The generalization is uncomfortable: allocation-reduction and parallelizability pull in opposite directions on scratch buffers, so hoisting one out of a loop should be paired with asking whether that loop could otherwise fan out.

BIT-IDENTITY was free here and worth noting why: column c writes only out[i*cols+c], reads only its own scratch, and its substitution order is untouched - no accumulation crosses columns. The tolerance-0 goldens written for these three functions many iterations ago held unchanged, which is the payoff for having written them then.

A SMALL-SIZE FALSE REGRESSION, the fourth distinct benchmark-reading failure this session. LUSolve_128x1 reported a 2.09x regression at benchtime=5x - 115 microseconds of total measurement on a 20us benchmark. At 20000x it is 20.1us against 19.4us with overlapping ranges. The pattern across all four: phantom regression from one sample, inverted sign from a bimodal min-of-N, spread on a path the change does not reach, and now insufficient total measurement time on a fast benchmark.

STILL OPEN, from the same sweep: rl DQNLearn and PPORollout are SLOWER at 12 cores than at 1 (71.4us vs 50.2us, and 1.03ms vs 0.74ms). That is the opposite of everything else found so far and deserves its own investigation - but PROC-PROFILE-PARALLEL-CONDVAR-001 and the GBM rejection say to establish it by wall-clock first, which the sweep has now done.

## R-01KYR63H61E179XJFQ7QWQ6VA8 CORRECTION: rl is not anti-parallel. The slower-at-12-cores reading was a fifth insufficient-benchtime artifact
kind: research
state: draft
created: 2026-07-30

CORRECTS the open item left in R-01KYR5ZS7RF7H, which reported rl DQNLearn and PPORollout as SLOWER at 12 cores than at 1 and flagged it as deserving investigation.

THE ORIGINAL READING came from the GOMAXPROCS sweep at benchtime=5x: DQNLearn/batch32 71.4us at P12 against 50.2us at P1, and PPORollout 1.03ms against 0.74ms. On a 50-70us benchmark, five iterations is roughly 300us of total measurement.

RE-MEASURED at benchtime=2000x, two rounds each:
  PPORollout        P12 659-665us   P1 688-751us   P12 slightly FASTER
  DQNLearn/batch32  P12  64- 66us   P1  62- 64us   flat, ranges nearly overlap
So neither path is anti-parallel. Both are approximately flat - parallelism neither helps nor hurts - which is what a dispatch-bound path on tiny tensors looks like. There is no pathology to investigate.

THE METHOD FAILURE IS THE POINT, and it is the fifth distinct instance this session, all from the same root: reading a ratio before checking that the measurement was long enough or that the arms differ in the way assumed. The five are a phantom 1.73x regression from a single sample (Mixtral prefill), an inverted sign from a bimodal min-of-N (DeepSeekV2), a spread on a code path the change never reached (SVC n=1000), insufficient total time on a 20us benchmark (LUSolve_128x1), and now insufficient total time inside a diagnostic SWEEP.

THE SWEEP ITSELF NEEDS THE SAME DISCIPLINE, which I had not applied to it. PROC-BENCH-TOTAL-TIME-001 was cast about A/B comparisons; a GOMAXPROCS sweep is an A/B in disguise - the two arms are core counts - and its cheapness was exactly what made me run it at benchtime=5x across dozens of benchmarks. The sweep is still the right instrument: it correctly found SoftmaxRegression at 0.99x, SVC at 0.97x and the entire linalg package at 1.00-1.10x, all three confirmed and all three fixed for real wins. But a sweep entry is a HYPOTHESIS to re-measure, never a result, and the ones worth acting on are the large clear signals rather than the marginal ones.

NO ACTION on rl. Its two flat paths were already optimized earlier this session (PPO rollout 1.59x, DQN learn 1.35x) by removing dispatch waste rather than by adding parallelism, which is the correct treatment for a dispatch-bound path and is why they are flat rather than slow.

## R-01KYR68YJYF8ZRYA80GR64CFYF GOMAXPROCS sweep completed across all four owned packages: two real finds, two clean negatives, and one invalid application
kind: research
state: draft
created: 2026-07-30

Closes the sweep line opened in R-01KYR4Z7SWFTC. Measured on darwin/arm64 M2 Pro, go1.26.5, at benchtimes giving tens of milliseconds per arm per PROC-SWEEP-IS-HYPOTHESIS-001.

FINAL RESULTS, parallel speedup from GOMAXPROCS=1 to 12:
  classic  ForestFit 7.93x, KNNPredict 5.87x, DBSCAN 5.67x, GMMFitFull 3.51x, GBM 1.53-2.79x
           SoftmaxRegressionFit 0.99x -> FIXED, 1.96x
           SVCFit/n4000_rbf     0.97x -> FIXED, 1.27x (Amdahl-bounded by sequential SMO)
  linalg   EVERY benchmark 1.00-1.10x -> FIXED, up to 7.87x on the solve phase
  vision   MLPMixer 1.85/4.07x, Swin 1.68/3.42x, ViT 2.14/4.32x (perimage/batched) - all parallel, NO finds
  rl       PPORollout and DQNLearn flat at ~1.0x, correctly - dispatch-bound on tiny tensors,
           already optimized earlier by removing dispatch waste rather than adding parallelism
  nlp      realistic geometry: MixtralPromptPrefill 4.20x, QuantMamba2Prefill_256 5.63x - all parallel

TWO CLEAN NEGATIVES ARE PART OF THE RESULT. vision has no flat path, and its perimage-versus-batched gap turned out to be caller-side: the per-image loop lives in the BENCHMARK, so fanning out across images is the user job and not a library change. rl is flat rather than slow, which is the correct end state for a dispatch-bound path.

ONE INVALID APPLICATION, and this is the durable limitation. Sweeping the twelve-architecture quant matrix reported every path at 1.04-1.16x, which looks like a package-wide serial bottleneck. It is an artifact: those fixtures are deliberately two layers at dim 32 - I built them that way, and recorded that they measure per-layer ALLOCATION COUNT, which is geometry-independent - so no backend op is large enough for the parallelism under test to engage. Re-running against benchmarks with realistic geometry gave 4.20x and 5.63x. THE SWEEP ONLY MEANS SOMETHING ON BENCHMARKS WHOSE TENSORS ARE BIG ENOUGH FOR THE PARALLELISM UNDER TEST TO ENGAGE, and a fixture set optimized for cheap deterministic allocation counting is precisely the wrong input. The failure is subtle because the flat numbers are perfectly reproducible - they are just measuring the fixture, not the code.

SCORECARD FOR THE INSTRUMENT: three large signals, all confirmed on re-measurement, all yielding real wins (1.96x, 1.27x, 7.87x). Two marginal signals, both artifacts of insufficient benchtime (rl). One whole-package flat reading that was a fixture artifact (nlp). The large-signal discipline in PROC-SWEEP-IS-HYPOTHESIS-001 is what separated them.

## R-01KYR87326EZ6929J966A3A8S4 linalg is done: 5.5x to 9.0x across the package, and the profile-reparallelize loop terminated on its own
kind: research
state: draft
created: 2026-07-30

Completes the linalg arc opened in R-01KYR5ZS7RF7H. Measured on darwin/arm64 M2 Pro, go1.26.5, three interleaved rounds per change, all ranges disjoint.

THE PACKAGE AT n=768, cumulative over three changes:
  LUSolve    549ms ->  61ms   9.0x
  Lstsq     1325ms -> 214ms   6.2x
  Inverse    676ms -> 112ms   6.0x
  CholSolve  629ms -> 114ms   5.5x
It measured 1.00-1.10x from one core to twelve when the GOMAXPROCS sweep found it.

THE METHOD WAS A LOOP THAT TERMINATED ITSELF. Each change made the next bottleneck dominant and the profile named it: parallelizing the solve phase raised LU Factor from 18% of CPU to about 60% of wall clock; parallelizing that left householder at 78% of Lstsq wall and cholFactor at 46% of CholSolve wall; parallelizing those leaves nothing above the solve phase, which is already parallel. Three iterations, each target selected by the previous fix rather than by search. Worth noting the arithmetic that makes a CPU profile usable here: total CPU divided by wall clock gives average parallelism, and a serial block share divided by that ratio gives its true wall share - which is how an 18% CPU entry was correctly read as 60% of wall.

THE AXIS WAS DIFFERENT EACH TIME and was the whole content of each change. Solve: over right-hand-side COLUMNS. LU elimination: over ROWS of the trailing submatrix. Householder: over COLUMNS of the trailing submatrix. Cholesky: over ROWS below the diagonal. In every case the accumulation axis stayed inside and untouched, per PERF-REDUCTION-AXIS-DECIDES-001, which is what made all four bit-identical rather than merely close.

ALLOCATIONS ARE THE HONEST COST. Each factorization opens one pooled region per step, so n=768 costs roughly a thousand extra allocations per call - Lstsq 1586 to 2613, CholSolve 820 to 1964, Inverse 826 to 1850. Against 475ms saved on Lstsq that is about 31us of allocator work, a clearly favorable trade, but it is a real regression on the resource axis and is reported as one rather than omitted. Two mistakes were caught by measuring it: spawning a worker set per step instead of using the pool cost 17463 allocations (21x), and passing a closure on the serial branch cost one allocation per step. Both are now rules.

SINGLE-CORE IS PARITY OR BETTER EVERYWHERE, with IDENTICAL allocation counts at GOMAXPROCS=1 and on small inputs, because every serial branch is written out inline. That was not free - it is duplicated loop bodies - and it is what makes these changes qualify under the constraint that an optimization improve things across systems rather than only on a twelve-core host.

NO PERFSCAN RULE for the factorization pattern. The shape - a sequential outer step whose inner update is independent - is real and recurred four times, but deciding independence requires knowing which array elements each iteration reads and writes, which is alias and dependence analysis rather than syntax. The tool that finds these is a GOMAXPROCS sweep followed by a profile, both already recorded as rules.

## R-01KYR8FT2KEP0RNM3JF7H5QED8 Correction to the GBM rejection: the per-chunk gate failed because my CONSTANT was 12x too high, not because the criterion was wrong
kind: research
state: draft
created: 2026-07-30

SHARPENS R-01KYR494K5ESB, which rejected a per-chunk work gate for GBM feature fan-out after measuring it 1.88x slower on GBMFit and 1.20x slower on GBMHist_exact_80k. That record attributed the failure to the CRITERION being wrong - total work versus per-chunk work. A later profile shows the real cause, and it is more mundane and more useful.

WHAT THE GATE ACTUALLY DID. It required d*n >= histParThreshold * workers, which at 1<<15 and 12 workers is 393216. With d=20 that means n >= 19661 samples per node. BenchmarkGBMFit uses n=2000 TOTAL, so every node fell below the bar and the gate disabled parallelism for the ENTIRE fit - hence 1.88x slower. GBMHist_exact_80k has 80000 rows, so only its deepest nodes were disabled, hence a milder 1.20x. The measurements were right; my reading of them was not.

THE CRITERION IS NEARLY A NO-OP HERE, which is the actually useful finding. The existing gate is d*n >= 1<<15, so n >= 1638 at d=20. A per-chunk gate with a correctly scaled constant asks for about 2 features times n, landing within a factor of two of the same place. At these shapes the two framings are a distinction without a practical difference, so there was never much to win, and the existing constant is already well calibrated for this caller.

WHAT THE PROFILE CONFIRMS. GBMHist_exact_80k spends 79 percent of CPU samples in pthread_cond_wait, cond_signal and usleep at 3.0x average parallelism - pprof reports that ratio directly as 300 percent of one core. Under PROC-PROFILE-PARALLEL-CONDVAR-001 that is about nine of twelve workers idling, which is what a 3x-parallel program looks like when sampled this way. GBM opens a region per tree node, roughly 750 for 50 estimators, so the barrier traffic is real - but measurement already established that REDUCING the region count makes it slower. The only remaining lever is a cheaper barrier, which means changing internal/parallel, shared with backend and outside my lane.

NO ACTION on GBM. The correction matters because the original rejection would discourage a future per-chunk gate on principle, when the honest statement is narrower: the criterion is fine, the constant must be derived from the caller's actual node sizes rather than multiplied by the worker count, and for this caller it would change almost nothing.

## R-01KYR8GNARE0BB53XAQ7N8MQCJ Two targets scoped out with reasons: ForestFit allocations are in the user's lane, attnReconstructed has no benchmark
kind: research
state: draft
created: 2026-07-30

Recorded so neither is re-derived. Both were located this iteration by applying PROC-PROFILE-WALL-SHARE-001 and PROC-BENCH-COVERAGE-NULL-001 to the remaining large numbers in my lanes.

FORESTFIT IS THE LARGEST ALLOCATION SITE LEFT IN classic AND IS OUT OF LANE. 352008 allocations per fit, roughly 40x the next-highest benchmark in the package. An allocation profile puts cartBuilder.buildIdx at 99.11 percent of them, 2101692 objects. That function lives in classic/tree.go, which the user works on in parallel and where main has three recent perf commits (incremental Gini, hoisted class counts, fused leaf histograms). Not touched. Anyone picking this up should confirm the lane is free first: the win is likely large and purely on the resource axis, since ForestFit already scales 7.93x from one core to twelve.

ATTNRECONSTRUCTED IS THE ONE PS6018 CANDIDATE I LEFT UNFUSED AND CANNOT VALIDATE. It is the second per-head loop in nlp/quant_deepseekv2.go, deliberately skipped when the absorbed path was fused, and PS6018 correctly reports it as the only surviving hit among the three files fused that round. A panic probe establishes the coverage gap precisely: it is reached by the DeepSeekV2 tests but by NO benchmark, including the twelve-architecture matrix, which takes the absorbed path instead.

The selector is per-BLOCK rather than per-config: b.WkvB != nil chooses reconstructed, nil chooses absorbed. So covering it needs a GGUF fixture whose blocks carry a kv_b weight, not a benchmark flag - which is why the arch matrix misses it. Under the standing constraint that only what is verifiable here may ship, the fusion is blocked on building that fixture, and the fixture is the larger half of the work.

WORTH STATING PLAINLY: the two reconstructed and absorbed paths mean this file has two per-head loops with the same movement-only fusion opportunity, and only one of them is exercised by any benchmark. That asymmetry is exactly what PERF-GATE-IMPLIES-BLIND-SPOT-001 predicts - the path excluded from the optimization was also the path excluded from measurement.

## R-01KYR93X7EFGBT100ET1AX4NK4 attnReconstructed fused after building its fixture: -16.6% allocs, and a per-FILE branch is the hardest kind of coverage gap to notice
kind: research
state: draft
created: 2026-07-30

CONSUMES the second item scoped out in R-01KYR8GNARE0B. Measured on darwin/arm64 M2 Pro, go1.26.5, three interleaved rounds at benchtime=100x.

MEASURED:
  ReconstructedDecode   5767 -> 4807 allocs  -16.6%   301-307us -> 272-284us  1.11x
  ReconstructedPrefill  1203 -> 1011 allocs  -16.0%   time parity
  QuantArchDecode/DeepSeekV2 (absorbed) unchanged at 4686 allocs
Allocation counts identical every round. The unchanged absorbed figure is the isolation check: it proves the edit landed only on the branch intended.

THE COVERAGE GAP WAS PER-FILE, WHICH IS THE HARD CASE. Which of QuantDeepSeekV2's two per-head loops runs is decided by the loaded GGUF: split-form keys (attn_k_b/attn_v_b) give the absorbed operator, the legacy unsplit key (attn_kv_b) gives the fused reconstruction and takes attnReconstructed. Not a config flag, not a size threshold, not a dtype - a property of the input FILE. So no amount of parameterizing the existing benchmark would have reached it, and the twelve-architecture matrix builds the split form throughout. The path was covered by tests and by zero benchmarks.

That is worth separating from the earlier coverage failures. A size below a threshold (SVC n=1000) or a float-versus-quant model (Cohere) are visible in the benchmark's own arguments. A branch on file CONTENT is only visible by reading the loader, and the symptom - a rule reporting one stubborn candidate - looks like a deliberate exception rather than a blind spot. PS6018 flagging it every run is what kept it from being forgotten.

BUILDING THE FIXTURE WAS THE WORK; the fusion was mechanical and identical to the absorbed one. The existing legacyQuantDeepSeekV2GGUFBytes already produced the unsplit form for a test, so widening it to testing.TB and writing two benchmarks over it was cheap - but only because someone had written that helper. The benchmarks assert WkvB is non-nil, so they fail loudly rather than silently degrading into measuring the absorbed path, which is the failure mode PROC-BENCH-COVERAGE-NULL-001 describes.

ONE ALIASING HAZARD CHECKED RATHER THAN ASSUMED. The value buffer is reused across heads and handed to appendKV, which stores into the KV cache. rowBuf.Append copies its row argument via copyRows and returns a view of its OWN backing, so no head's value block can alias another's. Had it retained the argument the cache would have been corrupted silently, with every head seeing the last head's values - and the parity test would likely have caught it, but only after the fact. Reading the callee is cheaper than debugging that.

PS6018 now reports ZERO in quant_deepseekv2.go, having tracked the file from two findings to one to none across three iterations.

## R-01KYR9KZZ6FSKBGYG9VEXNDBMQ Profiling allocated BYTES instead of object count found a target nine iterations of object-count profiling had walked past
kind: research
state: draft
created: 2026-07-30

Measured on darwin/arm64 M2 Pro, go1.26.5, three interleaved rounds. Allocation counts deterministic.

THE AXIS WAS THE FINDING. Every allocation profile in this sweep so far used alloc_objects. Switching one to alloc_space put QuantMamba2Mixer.forward at 61% of bytes for a 256-token prefill and rows2D at 206MB across the run, roughly 41MB per op and 38% of the per-op footprint. rows2D had never appeared near the top of an object-count profile because its allocations are FEW AND LARGE: one per row, r+1 per call. Object count and byte volume rank differently, and a sweep that only ever asks for one of them has a systematic blind spot the size of the other.

SHIPPED: rows2D sub-slices a single backing array instead of allocating per row, each row capped at its own length so an append cannot reach into the next. Contract unchanged - every row is still an independent copy.
  QuantMamba2Prefill_256      2023 -> 1516 allocs  -25%   108.57MB -> 104.90MB  -3.4%
  QuantMamba2PrefillQ8_0_128  1233 ->  980 allocs  -20%     4.77MB ->   4.72MB
Time parity in both, ranges overlapping.

THE BYTE DROP WAS NOT THE PREDICTED EFFECT and is the larger half of the value. The change targeted allocation COUNT; bytes fell 3.67MB because 256 separate per-row allocations each round up to a size class and waste the remainder, while one array of r*c rounds once. Per-row size-class waste is invisible in both a count profile and a byte profile - it only shows up in the B/op delta of an A/B - which is a third thing worth measuring on any change that consolidates many small allocations into one.

DELIBERATELY NOT DONE, and the reason is a contract rather than effort: returning VIEWS into the tensor storage would remove the remaining bytes, which is where the footprint actually is. rows2D has 30 call sites and that change makes every one of them alias its input. In the mixer the rows are only sliced into read-only views, so it would probably be safe there - but probably, across 30 sites, is not a basis for changing a helper's aliasing contract. The alternative is auditing each site for mutation, which is a task rather than an edit. Recorded in the code at the call site so the next reader inherits the analysis instead of redoing it.

## R-01KYRA0Y9PE81B8VQGQ6PQCXND The alloc_space sweep found a second target immediately: GBM subsample buffer, up to -55% of a fit's bytes
kind: research
state: draft
created: 2026-07-30

CONSUMES PROC-PROFILE-BOTH-ALLOC-AXES-001 one iteration after casting it. Measured on darwin/arm64 M2 Pro, go1.26.5, three interleaved rounds; byte and allocation counts deterministic.

MEASURED:
  GBMHist_exact_20k  20.25MB -> 12.22MB  -39.7%
  GBMHist_hist_20k   15.16MB ->  7.13MB  -53.0%
  GBMFit              5.95MB ->  2.71MB  -54.5%
Time parity everywhere, ranges overlapping - a footprint change, not a throughput one. The mechanism confirms to three digits: 50 rounds x 20000 ints x 8 bytes is 8MB and the observed drop is 8.03MB on both 20k benchmarks.

THE SWEEP PAID OFF IMMEDIATELY, which is the point worth recording. Running alloc_space over four benchmarks in my lanes took one command each and produced two actionable sites - nlp rows2D last iteration and GBM subsampleIdx this one - both invisible in the alloc_objects profiles that had been run repeatedly over the same code. subsampleIdx is 31% of a fit's bytes and about 50 of its 6700 allocations, so a count profile ranks it near the bottom while a byte profile ranks it second.

SAFETY WAS CHECKED IN BOTH GROWERS, not the one the benchmark exercises. histBuilder.grow copies idx into idxbuf because its partition works in place; gbmBuilder.fit only reads idx to filter its presorted columns. GBMHist_exact_20k runs the exact grower only, so checking that one and shipping would have left the histogram path unverified for a change that alters both.

RNG DETERMINISM IS THE NON-OBVIOUS PART of the bit-identity argument. Reusing the buffer is only safe because idx is refilled with the identity permutation before each shuffle - the same draws must act on the same starting order. A version that skipped the refill would consume identical draws and select a different sample, and the golden tests would catch it, but the reason is worth stating rather than leaving to the test.

A FALSE TARGET REJECTED IN THE SAME PROFILE. newGBMBuilder is the largest byte site at 53%, and it is a ONE-TIME presort per fit rather than per round - already hoisted outside the boosting loop. The benchmark constructs a fresh model each iteration, which makes a one-time cost look recurring in a profile aggregated over iterations. There is no reuse to exploit; the fix would have been across calls that do not exist. Reading the call site before believing the profile is what separated it from the real target sitting just below it.

## R-01KYRADWTWFGSVJZ2QWTD3TGHQ CORRECTION: the three alloc_space wins are three DIFFERENT patterns, not one. The rule I built from them found none of them
kind: research
state: draft
created: 2026-07-30

CORRECTS a claim made in the KNN heap commit message, which asserted the three alloc_space findings shared one shape: a container left at its zero value and grown by append where the final size is known at construction. That was wrong, and building the rule is what proved it.

WHAT THE THREE ACTUALLY ARE:
  nlp rows2D        out := make([][]float64, r) with LENGTH r, rows assigned BY INDEX. The fix was consolidating r separate per-row make calls into one backing array - an allocation-count consolidation, no append anywhere.
  GBM subsampleIdx  idx := make([]int, n) with LENGTH n, filled by index. The fix was REUSING one buffer across boosting rounds - a lifetime change, not a sizing one.
  KNN knnHeap       items left nil in a struct literal and grown by append to k. The fix was preallocating to the known bound - the only one matching the claimed shape, and even it is a zero-value struct FIELD rather than a local slice.
Three distinct interventions: consolidation, reuse, preallocation. What they share is only the axis that found them (alloc_space) and the fact that alloc_objects had walked past all three.

HOW THE ERROR SURFACED. I implemented PS6020 append-grown-known-size for the claimed shape, then replayed it against the pre-fix source of all three instances per PROC-SCANRULE-REPLAY-001. It found ZERO of them while reporting 13 candidates elsewhere in the tree. A rule that fires thirteen times and matches none of its motivating cases is not a rule for those cases; it is a rule for something else that happens to be syntactically nearby.

REVERTED, and the reason is PROC-NO-RULE-FROM-UNVALIDATED-001 rather than the miss itself. The 13 candidates might well be real - a nil slice grown by append under a known bound genuinely costs log2(n) allocations plus overshoot - but I have no MEASUREMENT for any of them in this repository. Shipping the rule would have asserted that shape is a defect here on the strength of three wins that were not instances of it. The honest move is to keep the rule out until one of the 13 is measured; the shape is recorded here so that is cheap to do later.

THE PROCESS LESSON, which is why this is worth a record rather than a silent revert: the pull to generalize is strongest right after a win, and the standing mandate to turn generalizable findings into scan rules makes it feel obligatory. Two guards caught it - replay against the motivating source, and the requirement that a rule cite a measurement. Both were cast earlier in this session after similar failures, and both fired exactly as intended. The pattern-recognition step between measurement and rule is where the error lives, and it is not covered by either guard: nothing forced me to check that the three cases were the SAME case before naming their shape.

## R-01KYRANC8DFNERJP95C3JRQ9NY PS6020 rejection is now COMPLETE, not merely unvalidated: every append-grown slice in my lanes fails the preconditions
kind: research
state: draft
created: 2026-07-30

COMPLETES the rejection opened in R-01KYRADWTWFGS, which reverted PS6020 append-grown-known-size for lack of a measurement on any of its 13 candidates. This iteration checked the candidates instead of the rule, which is the step PROC-NO-RULE-FROM-UNVALIDATED-001 asks for.

EVERY CANDIDATE IN MY LANES FAILS A PRECONDITION, and the failures fall into three kinds:
  CONDITIONAL APPEND - the loop bound is an upper estimate, not the count, so reserving it over-allocates whenever the guard rarely fires. nlp/blt.go:178 (appends on an entropy threshold), diffusion_lm.go:177 and :279 (only masked positions), guided.go:220 (only matching instruction ops), coconut.go:334 (early-returns on an error), classic/dbscan.go:279.
  NO KNOWN BOUND - nlp/bpe.go:244 gpt2Split grows over a LAZY SEQUENCE of pre-tokens whose count is not available anywhere; and its own comment records that the hot path, Tokenizer.Encode, already ranges the sequence directly to avoid the slice, after a profile put it at 36% of encode's allocations. Someone had already done the real fix here.
  ALREADY REUSED OR NOT HOT - classic/dbscan.go:212 declares its DFS stack OUTSIDE the per-point loop, so it grows once and is reused across every cluster expansion. dbscan.go:170's tree path appends inside ballTree.radius and RETAINS each result as neighbors[i], so there is no buffer to reuse - each point's neighbour list must persist independently.
The one unconditional case, diverse_beam.go:161, has a bound that is the sum of len(grp[g]) over groups - computable but not an identifier in scope, so the fix is a preliminary counting pass rather than a capacity argument.

SO THE REJECTION IS NOW EVIDENCE-BACKED. Last iteration's statement was the honest but weaker one: no measurement, therefore no rule. The stronger statement is that the shape does not occur in a harmful form in classic, nlp, linalg, vision or rl - and the reason is systematic rather than accidental. An append inside a loop is USUALLY conditional; that is why the append is there instead of an indexed write. Where it is unconditional the author generally already knows the size and has written make with a capacity. The KNN heap was the exception because its container is a struct field initialized by a composite literal, where the zero value is easy to leave in place.

WHAT WOULD CHANGE THIS. A rule that flagged only the struct-field form - a zero-value slice field on a type whose constructor knows the bound - would have matched the one real instance and none of the six conditional ones. That is a narrower and probably sound rule, but it has exactly one measured instance in this repository and PROC-RULE-FROM-N-INSTANCES-001 now requires more than that before naming a shape. Recorded here so a second instance, if it appears, arrives with the analysis already done.

## R-01KYRBBDHRE9HVYKQTDVQDXP18 BPE token slice reservation: allocations -65%, and the PS6020 rule now has two measured instances against six non-instances — precision 25%, still no rule
kind: research
state: draft
created: 2026-07-30

Measured on darwin/arm64 M2 Pro, go1.26.5. Deterministic on the allocation axes.

SHIPPED: BPEGGUFEncode 34 to 12 allocations, -64.7%, and 3.22MB to 2.55MB, -20.8%. Decode unchanged, an Encode-only change. A line-level alloc_space profile put out = append(out, id) inside bpeInto at 99.93% of the bytes - 616MB across 200 iterations - because ids grew by doubling from empty on every call. bpeInto already threaded its parts scratch and output slice across pieces; only the initial capacity was missing, which is why the site had survived several rounds of work on this exact function.

NO TIME CLAIM, and the discipline is the point. Five rounds at benchtime=1500x give 4.370-4.484ms against 4.441-4.527ms. The ranges overlap slightly and the within-arm spreads, 2.6% and 1.9%, are comparable to the 1.6% gap - so the spread check fails and the honest report is deterministic allocation wins with time at marginal-to-parity. An earlier three-round run at 300x showed non-overlapping ranges and a 3.5% gap, which would have supported a time claim; raising benchtime dissolved it.

THE RULE CALCULUS, which is why this closes rather than reopens PS6020. There are now TWO measured instances of reserve-capacity-at-declaration: the KNN heap, an exact bound k on a zero-value struct field, and this one, an estimate len(text)/3 on a local slice. Same intervention, so PROC-RULE-FROM-N-INSTANCES-001 is satisfied on that axis. But the same lanes contain SIX sites where the shape appears and preallocation is wrong - conditional appends whose loop bound is an upper estimate that would over-reserve (R-01KYRANC8DFNE). A rule flagging the shape would run at 25% precision: two true positives against six known false ones. That is worse than the advisory rules already in the set, and PERF-SCANRULE-EMPTY-001's sibling argument applies - a rule whose hits are mostly wrong trains readers to skip it.

WHAT WOULD MAKE IT SHIPPABLE is a predicate separating an unconditional append from a guarded one, which the AST can see, combined with a bound available at the declaration. The six false positives are all guarded, and both true positives are effectively unguarded - the KNN heap appends whenever below k, and bpeInto appends on every piece unless a vocab miss coincides with no unk token. So the precision problem is solvable; it needs the guard predicate implemented and re-validated against all eight sites, which is a task rather than an aside. Recorded with the site list so that work starts from evidence.

## R-01KYRBHDN1FMST22Q0X2X4HHQG PS6020 is definitively not AST-ruleable: the deciding property is a guard's HIT RATE, and my proposed predicate would have excluded both true positives
kind: research
state: draft
created: 2026-07-30

CLOSES the PS6020 line for good, and corrects two claims I recorded one iteration earlier in R-01KYRBBDHRE9H. That record proposed shipping the rule with a guard predicate on the grounds that both measured true positives are effectively unguarded while all six false positives are guarded. Both halves were wrong, and checking before building is what caught it.

WHAT THE TRUE POSITIVES ACTUALLY LOOK LIKE:
  knnHeap.consider - the append sits inside if len(h.items) < h.k, so it IS guarded; and it is not inside a loop within that function at all. The loop lives in the CALLER, searchKNN, which calls consider once per candidate point. The shape spans two functions, so a rule scanning one function body cannot see it regardless of predicate.
  bpeInto - the append sits inside if id, ok := t.vocab[...]; ok, with an else-if branch for the unk token. Guarded, in a loop, and therefore a member of the exact class I had classified as the false-positive class. Preallocating still cut allocations 65 percent and bytes 21 percent.
So the proposed predicate would have produced ZERO true positives and correctly excluded the six - precision undefined, recall zero. Exactly wrong.

THE REAL DECIDING PROPERTY IS THE GUARD'S HIT RATE, which no parser can read. bpeInto's guard is a vocabulary hit, which is the normal case, so the loop bound is a good capacity estimate. The six rejected sites guard on an entropy threshold, a mask test, an instruction-op switch and an early error return - conditions that fire on a minority of iterations, where reserving the loop bound over-allocates. Same syntax, opposite conclusion, decided by a runtime frequency. That is not a shape; it is a measurement.

CORRECTED PERF-APPEND-USUALLY-CONDITIONAL-001, which I had cast with the wrong reasoning: it now says to estimate how often the guard fires rather than whether one exists. As originally worded it would have talked a future reader out of the bpeInto win.

THE META-OBSERVATION, third instance this session. The error was in the pattern-recognition step between a measurement and a rule, not in the measurement and not in the rule's implementation. PROC-RULE-FROM-N-INSTANCES-001 asks whether the instances are the same intervention - they were, both reserve-capacity-at-declaration. What it does not ask is whether the PREDICATE separating them from non-instances actually holds, and that is where all three failures landed. The cheap guard is to state the predicate and check it against every catalogued site BEFORE writing the detector; here that took two greps and would have saved building the rule twice.

## R-01KYRC2ATJEVJVY8TDAPK6N162 PS6018 at nlp/eagle.go:202 rejected on leverage, and the attempt to encode that as a predicate failed the same way PS6020 did
kind: research
state: draft
created: 2026-07-30

REJECTS the last open PS6018 candidate in my lanes, and records that the second attempt in two iterations to refine a perfscan check with a syntactic hotness proxy failed, for the same underlying reason.

THE CANDIDATE. PS6018 fires on EagleLoss because the function dispatches three pure data-movement ops with no fused raw-storage path: a row slice of hidden for rows 0..n-2, a second row slice of hidden for rows 1..n-1, and a row slice of pred for rows 0..n-3. All three are contiguous row ranges of a rank-2 tensor, so fusing them is bit-identical by construction, exactly as the check says.

WHY IT IS STILL A REJECT. The three dispatches execute exactly three times per call. They are not in a loop, and EagleLoss is called once per training step, where it is dominated by the draft head's two dim-by-dim matmuls and by a vocab-wide classification matmul. Three tensor allocations and three dispatches against a vocab-sized GEMM is noise. Separately there is no benchmark in the package that calls EagleLoss at all, so nothing here is measurable on this host without first writing one, and the estimated ceiling does not justify that.

FIRST PREDICATE ATTEMPT, REJECTED BEFORE IMPLEMENTATION: suppress PS6018 inside loss functions, on the theory that the recommended fix gates the fused arm on an untaped context while a loss only runs under a tape, making the fused arm dead code. Checked against the catalogue as the new predicate-first rule requires. It failed twice over. Of 26 PS6018 sites tree-wide exactly one is a loss function, so the class buys one site; and EagleLoss is in fact called untaped in its own reference tests, so the fused arm would not be dead code. Premise false, yield negligible.

SECOND PREDICATE ATTEMPT, ALSO REJECTED: rank or suppress by loop nesting, on the theory that a movement cluster inside a loop body is hot and one at function top level is not. Checked against the three shipped PS6018 wins - partial-RoPE, Gemma2 capped attention, DeepSeekV2 absorbed attention. All three have their movement cluster at function top level, structurally identical to the rejected candidate. The predicate would have excluded every shipped win. Zero recall again.

THE ACTUAL DISCRIMINATOR, and it is not syntax. What separates the three wins from this reject is that each winning function is entered once per layer per decoded token, while EagleLoss is entered once per training step. That is a call frequency: cross-function, and a runtime fact. This is the second consecutive check whose precision ceiling is set by call frequency rather than by shape - PS6020's was a guard's hit rate. The conclusion is not that perfscan needs a better proxy; it is that the tool is already correctly scoped as a candidate generator whose output line says to measure hotness before shipping. Attempts to push hotness into the AST should stop.

WHAT WOULD ACTUALLY HELP, recorded but not built: annotate each candidate with whether any benchmark in the tree reaches the enclosing function, which converts a flat list of 26 into the triage the workflow actually needs - measurable now versus needs a new benchmark first. That requires a name-based cross-package call graph, which is a real feature rather than a predicate tweak, and is a separate task.

## R-01KYRCQS8BEQS8EK3BG0EHACBX Declined a perfscan rule for the declined-reuse-buffer shape: one instance tree-wide, zero remaining after the fix
kind: research
state: draft
created: 2026-07-30

The DBSCAN neighbour-list win has a clean, AST-visible generalization, and it was still the wrong thing to build. Recording the arithmetic so nobody re-derives it.

THE CANDIDATE SHAPE. A callee advertises a reuse contract by taking a destination slice parameter and truncating it on entry (dst = dst[:0]), which means a caller can hand back the same buffer every iteration. A caller that instead passes a literal nil inside a loop declines that contract and pays a fresh growing allocation per iteration. Both halves are visible to a parser: the truncation-on-entry marks the contract, and a nil argument in a loop body marks the declined call. Precision would be high because the callee explicitly opts in.

WHY IT IS NOT WORTH BUILDING. Swept the tree for the contract first, as the predicate-first rule requires. Exactly one function advertises it - the ball tree's radius query - and its only nil-in-a-loop caller was the DBSCAN site just fixed. So the rule would ship with one historical instance and zero live ones, and would sit dormant until someone independently invents the same contract elsewhere. That is the PS6020 mistake with the sign flipped: there the shape had too little discriminating power, here it has plenty but nothing left to discriminate.

THIS IS THE THIRD CONSECUTIVE ITERATION where checking the predicate against the tree before writing the detector killed the detector, and each kill cost two commands. The check is now clearly cheaper than the build in expectation, not just in principle.

WHAT DID GENERALIZE was not a code shape but a measurement failure, cast as the benchmark-regime rule: assert the regime a benchmark's parameters put the code in, because a benchmark whose input silently makes the target path unreachable keeps reporting healthy numbers. That is the transferable lesson from this work, and it is a validation rule rather than a scan rule.

RESIDUAL, deliberately not done: a rule for the second half of the DBSCAN win - storing a value into a container when every read of that element is guarded by a condition, making the store dead - would need dataflow rather than shape matching, and perfscan is a parser with no type or flow information by design.

## R-01KYRCS4PDFAV8WY8BBP32MQEZ Vision forward-path survey: patchify is the one measurable lever; MAE and VLM are unmeasured and dispatch-bound
kind: research
state: draft
created: 2026-07-30

Delegated survey of vision/vit.go, mlpmixer.go, mae.go, vlm.go, vision.go, excluding swin.go. Findings below are the agent's, compacted; the measurement claims are ITS estimates and none is validated yet. Two structural facts I confirmed independently: the package has exactly three benchmarks (ViT, MLPMixer, Swin forward-batched), and NEITHER mae.go NOR vlm.go NOR vision.go is reached by any benchmark anywhere in the tree.

FRAMING. All five files are thin orchestration over nn.Linear, nn.LayerNorm and nlp.MHA; every FLOP-heavy loop is out of lane. What is in lane is the patch/pack/unpack glue. So the honest ceiling on the two benchmarked models is low single-digit percent, because they are GEMM-bound. MAE is the exception: its forward is roughly 0.75M MAC against about 130 backend dispatches, making it dispatch-bound, and nothing measures it.

MEASURABLE NOW, in descending leverage:
  patchify, three byte-identical copies at vit.go:206, mlpmixer.go:248, mae.go:385. Reads pixels one at a time through a closure returned by makeReader, widens f32 to f64, appends into a staging buffer sized n*C*p*p (loop-invariant, allocated per image), then makes a second full pass narrowing back into the output. The nest is C*size*size = the pixel count, 3072 per image and 24576 per batched iteration at bench geometry. Fix: drop the staging buffer, take the flat backing slice once, and exploit that the innermost run of length p is contiguous in BOTH source and destination, so f32-to-f32 and f64-to-f64 become copy() of p elements - 768 memmoves per image instead of 3072 closure calls plus 6144 conversions plus 3072 appends. Bit-identical on all four dtype pairs: no arithmetic, unchanged traversal order, and the f32-f64-f32 round trip is an exact identity. Covered by both existing ViT and Mixer benchmarks and by the two out-of-package benchcompare ViT benchmarks.
  The batched Forward wrapper at vit.go:269 and mlpmixer.go:329 slices, reshapes and patchifies per image then concatenates: 2B+1 dispatches and four passes over the batch pixels. Writing all images into one destination allocated once removes the dispatches and three of the four passes. Legal because patchify ALREADY severs the tape - it copies into a fresh tensor with no op node - so no gradient path to the input exists to lose. Covered by the batched sub-benchmarks only; the per-image sub-benchmark takes a different route.
  Mixer's per-image mean pooling at mlpmixer.go:362 slices each image out purely so the reduce can read it. One reshape to rank 3 plus one reduce over the middle axis replaces 2B+1 dispatches. NOT known bit-identical: whether reducing axis 1 of a rank-3 tensor accumulates in the same order as axis 0 of a rank-2 one is a backend property, and the existing parity test's tolerance is 1e-9 rather than exact, so it would not catch a genuine reassociation. Compare f64 storage exactly before accepting.

BLOCKED ON A MISSING BENCHMARK - all in MAE, and all cheap once one exists: gatherRows/unshuffleRows at mae.go:450 and 428 issue one slice dispatch PER TOKEN and then one concat, 128 slice dispatches per forward at S=64 with mask ratio 0.75, each allocating a one-row tensor to move D floats; a single row-gather op replaces each nest. Mask() at mae.go:373 rebuilds a seed-fixed permutation on every forward, so it is a pure constant of the model. Reconstruct at mae.go:521 patchifies the same image twice, once directly and once inside Encode. The mask being seed-fixed also means every image in a batch shares one keep/masked partition, so MAE batches with no ragged-length problem - the largest single lever here, and the same lever already booked for ViT.

REJECTED with reasons, so nobody re-derives them: the Mixer block's per-image transpose clusters look like the biggest dispatch site in the benchmarked path but cannot be improved, because the transpose op is rank-2 only with no batched permute, so slice-transpose-concat is already the three-pass minimum; doing it as a raw copy would sever the tape between the norm and the first projection and break gradients. Constructor-only helpers, Params() accessors, the CNN path in vision.go and the straight-line op chains in maskedMSE and the VLM projector are all cold or have nothing loop-invariant. Also noted, not a perf item: makeReader ignores the tensor offset, unlike the swin flat helpers which slice by offset and length - latent today because callers pass offset-0 tensors, but do not reuse that helper shape.

## R-01KYRCTQW2F20B945VVT544FFR Classic unswept-file survey: two candidates already shipped from it, GaussianNB row allocation is the remaining measurable one
kind: research
state: draft
created: 2026-07-30

Delegated survey of classic/dbscan.go, knn.go, naivebayes.go, linalg.go. Two of its candidates are already implemented and measured this session, which is the useful validation of the survey itself; the rest is compacted below with the agent's estimates marked as estimates.

CONFIRMED AND SHIPPED FROM THIS SURVEY. The KNN per-query vote allocations, independently found and fixed with per-worker scratch: allocations down 99.4 percent, bytes down 94.1 percent, and 2.1 percent faster (p=0.014, n=9). The survey predicted an allocation-metric win with no wall-clock effect; the small latency win came from the heap allocations in the spatial index, which the survey had excluded from its own scope. The DBSCAN neighbour-list reuse plus core-only retention, also shipped: allocations down 78.8 percent on both benchmark arms, bytes down 40 percent on the non-degenerate one.

THE SURVEY'S MOST VALUABLE OUTPUT was not a candidate but the observation that the DBSCAN benchmark's eps put the algorithm in a regime where the labeling phase never executed. I verified that directly - 0 clusters, 0 core points, all points noise - and it is now fixed and asserted. Cast as the benchmark-regime rule.

REMAINING, MEASURABLE. GaussianNB.jointRow at naivebayes.go:230 allocates its result slice per query row: one allocation per row against a Predict whose only other allocation is a single label slice. The covering benchmark sets n=4000, d=20, 3 classes, above the parallel threshold, and is the only case in scope where the covered hot path is entirely in-scope code, so roughly 4000 allocations collapse to about 12. The structural blocker is the same one KNN had: the parallel helper takes a body keyed by row index only, giving the body no worker identity, so no buffer can be hoisted without becoming shared state. The fix is the same - make the helper chunk-granular so scratch is allocated per job - and it needs the tail guard that a chunk-granular form requires but a per-row form silently tolerates. The existing body assigns every output entry unconditionally, so a reused buffer needs no clearing; that is the correctness hinge and belongs in the comment.
  cholSolve at linalg.go:15 allocates the factor row by row and reloads two row pointers inside an n-cubed-over-six inner loop. One flat backing array plus hoisting the row slices out of the innermost loop is bit-identical, same ascending accumulation. Honest denominator: the caller routes the normal equations through two matmuls, so the factorization is a minority of the time; the defensible claim is allocations and bytes, 66 to 1. The smaller OLS arm is below any measurable threshold - label it so rather than reporting a ratio.
  Flat arenas for the row-per-sample returns in both PredictProba paths and JointLogLikelihood, using three-index slices so rows cannot alias-append. Bit-identical, but nothing benchmarks any of those three entry points today.
  A cache-locality interchange in GaussianNB.Fit, which currently streams the whole row array once per feature per pass. Bit-identical because each fixed-feature summation stays ascending and only independent chains are reordered. No benchmark covers Fit at all.

REJECTED with reasons. Per-element tensor dispatch is absent from all four files - they operate on native Go slices, not tensors. No new parallelization is both legal and measurable: every hot loop the six benchmarks reach is already chunk-parallel. Parallelizing cholSolve is illegal on the reduction axis and the one legal axis is 65 wide. Parallelizing the NB fit passes over rows is illegal for the same reason; over features it is legal but only 20 units wide and unbenchmarked. DBSCAN's cluster expansion is inherently order-dependent, not an FP problem. The brute-force neighbour paths are unreachable above 16 points. The width-validation pre-passes are deliberately serial so the reported offending row index is deterministic. A wider unroll of the NB class loop would never execute, since every covering benchmark has 3 classes.

## T-01KYRCTTDCEAJ828V5P86X4RGS perf(vision): fuse patchify into a contiguous row copy across vit.go, mlpmixer.go and mae.go
kind: task
state: draft
created: 2026-07-30

SCOPE. vision/vit.go patchify (~line 206), vision/mlpmixer.go mixerPatchify (~line 248), vision/mae.go patchify (~line 385). Three byte-identical copies; change all three or none, and prefer factoring one shared helper in the package.

CURRENT SHAPE. Each obtains a per-element accessor closure from makeReader(img.Contiguous()), appends every pixel into a staging []float64 sized n*channels*p*p, then makes a second full pass narrowing or widening into the output tensor's backing slice. Per pixel that is one indirect call, one widen, one bounds-checked append, then one narrow and one store. The staging buffer's size is loop-invariant and it is allocated per image.

THE NEST is grid*grid*channels*p*p, which equals channels*size*size - the pixel count, independent of patch size. At the ViT and Mixer benchmark geometry that is 3072 per image and 24576 per batched iteration.

INTERVENTION. Delete the staging buffer and write straight into the output's backing slice, exploiting that the innermost run of length p is contiguous in BOTH source and destination: the source index is base + dx with base fixed per (py,px,c,dy), and the destination is append order. For f32-to-f32 and f64-to-f64 that makes each run a copy() of p elements - 768 memmoves per image instead of 3072 closure calls plus conversions plus appends. For the mixed dtype pairs keep an element loop but convert once directly rather than through float64. Keep img.Contiguous() and keep the existing unsupported-dtype error branch. Take the flat slice with an OFFSET-CORRECT helper: the swin flat helpers in vision/swin.go slice by offset and length, whereas makeReader indexes the whole storage and ignores the offset - reuse the swin form, not makeReader's.

BIT-IDENTITY. Bit-identical on all four dtype pairs. There is no arithmetic; the traversal order is unchanged; the f32-f64-f32 round trip the current code performs is an exact identity, and an f64-to-f32 narrowing happens exactly once either way. Verify against TestViTTorchParity, which holds an exact torch golden at 1e-9, plus the ViT and Mixer batched parity tests.

VERIFY. gofmt -l, go vet ./vision/, go test ./vision/ -count=1, and go test ./vision/ -race -run 'ViT|Mixer|MAE' -count=1. Then A/B with at least three interleaved alternations per PROC-INTERLEAVE-001: go test ./vision/ -run '^$' -bench 'BenchmarkViTForwardBatched|BenchmarkMLPMixerForwardBatched' -benchmem -count=3, stashing the change between arms, and report via benchstat. Check within-arm spread before claiming any ratio.

EXPECTED. Both models are GEMM-bound, so the honest expectation is low single-digit percent on wall clock plus one fewer allocation per image. Report whatever the measurement says, including no change; the mae.go copy is not covered by any benchmark, so state that its share is unmeasured rather than implying it was validated.

## R-01KYRDDNNZFN1T36TZ5B9G129Q PS6021 shipped: the fan-out-without-a-seam shape generalized where three prior rule attempts did not, and mutation testing found two defects in the first draft
kind: research
state: draft
created: 2026-07-30

Three wins landed this session shared one structural blocker, and unlike the last three rule attempts this one survived the predicate check and became a scan rule.

THE SHAPE. A parallel helper whose callback receives only a work index. The callback runs once per item, so a buffer the caller allocates inside it is allocated per item; hoisting it above the helper makes it shared mutable state every worker races on, and a receiver field is the same bug with a longer fuse. With no per-worker seam in the signature, per-item allocation is the only CORRECT option available. That is why these sites survive review indefinitely: the code is not careless, the interface is short a parameter. Measured after adding one - GaussianNB predict 1.28x with 99.2 percent fewer allocations, KNN predict 99.4 percent fewer allocations and 2.1 percent faster, DBSCAN fit 78.8 percent fewer allocations.

WHY THIS ONE GENERALIZED AND THE PREVIOUS THREE DID NOT. PS6020 and the PS6018 refinement both needed a runtime frequency - a guard's hit rate, a function's entry rate - which no parser can read. The declined-reuse-buffer shape was AST-visible but had one instance tree-wide and zero after the fix. This one is a property of a SIGNATURE, which is exactly what a parser does see, and it is a design invariant the codebase now follows in five or more places and violated in three. The distinguishing question, worth asking of any future rule proposal: is the deciding property syntax, or is it a frequency wearing syntax as a costume.

PREDICATE CHECKED FIRST, across all twenty-plus fan-out helpers in the tree: three historical true positives, one live true positive, zero false positives. Two floors do that work. A range callback is not reported, because the caller can allocate inside the chunk closure, which is per-chunk and therefore already per-worker - most helpers here have that shape, so reporting them would drown the real hits. A channel-creating helper is not reported, because that is a work-queue primitive whose callback IS the job, and every fix for this rule is built on top of one by passing a worker count as the job count; reporting the primitive would report the cure.

MUTATION TESTING FOUND TWO DEFECTS IN THE FIRST DRAFT, which is the part worth carrying forward. Breaking each predicate clause in turn must turn exactly one floor red. Two clauses failed that: a clause excluding helpers that take a scratch-constructor parameter turned out unreachable, because such a helper must pass the constructed value to its callback, so the callback already carries a scratch parameter and fails the index-only test - the clause was removed rather than left as implied coverage. And both post-fix floors were passing on the parameter COUNT alone, leaving the integer-type check unexercised; a floor with a single non-index parameter now gives it teeth, since one non-index parameter is a value visitor rather than an index fan-out. Neither defect was visible from a green test run. Nine tests total: two recall cases reproducing the pre-fix signatures, seven floors.

REMAINING INSTANCE, out of lane: logdetParallelIdx in autograd. Its callback takes a single index and its callers therefore cannot hoist. Not touched - autograd is the parallel worker's zone.

## R-01KYRE2HJ5EVXV51W8CKB6C0R0 Patchify fusion landed as a resource win, and an accidental null A/B established that the vision benchmarks cannot resolve a one-percent wall-clock effect
kind: research
state: draft
created: 2026-07-30

Implements the booked patchify task and closes it with honest numbers. Also records the measurement accident that turned out to be more valuable than the optimization.

WHAT SHIPPED. ViT, MLPMixer and MAE each carried a byte-identical patchify that read pixels one at a time through a closure, widened to float64, appended into a per-image staging buffer whose size is loop-invariant, then made a second full pass narrowing back out. The innermost run is contiguous at both ends - the source index makes consecutive dx adjacent, and the destination is filled in exactly that order - so each run of p elements becomes one copy. At benchmark geometry that is 768 memmoves per image instead of 3072 closure calls, 6144 conversions and 3072 bounds-checked appends, and the staging buffer disappears. The three copies now share one helper using the offset-correct flat accessors rather than the reader that ignores tensor offsets.

MEASURED across three interleaved alternations: allocations down 0.42 to 0.68 percent on all four benchmark arms, bytes down 0.49 to 2.81 percent on three of four, and NO measurable wall-clock change. The models are GEMM-bound, which the booked task predicted, so this is a resource win and is reported as one.

THE ACCIDENT WORTH KEEPING. The first A/B ran git stash over a file list including the newly created helper. Stash refuses untracked pathspecs, the subsequent pop reported no stash entries, and both arms therefore timed the NEW code. The run completed with plausible numbers and no visible error anywhere in the comparison output. That accidental null A/B reported a geomean sec/op delta of minus 1.39 percent with every individual comparison reading as insignificant. The valid run of the real change reported minus 1.15 percent. So the benchmark's own noise floor is larger than the effect, and without the accident the smaller number would have read as a modest win. Two rules cast: run a deliberate null A/B before believing any wall-clock delta under roughly five percent, and assert the base arm contains pre-change text before timing it whenever the change adds a file. Interleaving and min-of-N bound sampling error; neither says anything about a benchmark that cannot resolve the effect size at all.

BIT-IDENTITY was gated directly rather than through the model parity tests alone: comparison against an independent transcription of the documented layout at tolerance zero, across five geometries and both dtypes. The geometries pair channels above one with patch size above one deliberately, because with either equal to one several index terms collapse and a transposed layout would still compare equal; a non-square patch count and the single-patch degenerate case are included. Three mutations confirm the gate: swapping the channel and row loops, and corrupting each contiguous fast path individually. The last two matter most - they prove the test reaches the fast paths rather than passing through the fallback, which is the way a test like this usually goes vacuous.

STILL OPEN from the vision survey, in descending value: the batched forward wrapper's per-image slice, reshape and concatenate cluster, which is legal to fuse because patchify already severs the tape; the Mixer per-image mean pooling, which needs a backend reduce-order check before it can be called bit-identical; and everything in MAE, which remains reached by no benchmark in the tree.

## R-01KYRF4BTEEFKR4JMSFH2WW7YX Polyak fan-out shipped at 1.10x not 1.5-3x because the serial loop already saturated single-core bandwidth, and the parity test failed for a reason that was not the change
kind: research
state: draft
created: 2026-07-30

Implements the top candidate from the rl survey, and records two things more useful than the optimization.

WHAT SHIPPED. The Polyak target-network blend now fans out per parameter above 1<<15 elements. Legal because the blend carries no accumulation: every output index is written once from its own source element and its own previous value, so chunk order cannot move a bit; self-aliasing would be safe for the same reason. Measured minus 10.00 percent wall clock at p equals 0.015 over three interleaved alternations, against plus 66.67 percent allocations and plus 130 percent bytes. The allocation regression is the closure and fan-out bookkeeping for the two split parameters, and it is worth paying at this size: eight small allocations cost well under a microsecond including amortized collection, against 6.3 microseconds saved per call and two calls per training step. Below the threshold it would not be worth paying, which is why the threshold exists.

WHY 1.10x AND NOT THE ESTIMATED 1.5 TO 3x. The serial loop was already streaming 3.16 megabytes per call in 63.11 microseconds, about 50 gigabytes per second effective, which is near this host's single-core memory bandwidth. The parallel version reaches about 55.7. The loop was bandwidth-bound, not compute-bound, so additional cores had almost nothing left to contribute. An elementwise sweep of two reads and one write per element should be expected to behave this way, and estimating such a candidate from core count alone will overshoot every time. This is the useful calibration to carry forward: compute the effective bandwidth of the serial loop first, and if it is already near the per-core streaming limit, the ceiling is small no matter how many cores are available.

THE PARITY TEST FAILED FOR THE WRONG REASON, and that cost most of the effort here. The first version compared against a reference computed through Unravel and AtF64 into a separate slice, with the blend factor declared as a Go constant. It failed by exactly 1 ulp on the 65536-element parameters. Four experiments were needed to establish that the change was innocent: the closure form alone reproduces the original bit-for-bit; the closure with chunking disabled passes; a reference written in the pre-change flat-slice form with the factor as a variable gives ZERO differences across all 131841 elements; and const folding of one-minus-tau was ruled out by direct bit comparison, since the folded and runtime values are identical. What remained is FMA contraction differing between the two expression shapes, so the original test was measuring the compiler rather than the change. Cast as a rule: a bit-identity reference must transcribe the implementation's expression verbatim rather than being rewritten for readability.

BOTH GATES ARE MUTATION-VERIFIED. The corrected parity test turns red for a cross-chunk read and for pinning the FMA in the parallel arm alone. A separate test runs the same update eight times and requires bit-for-bit agreement, which catches a cross-chunk dependency that would appear as run-to-run variation rather than as a wrong answer on any single run.

FROM THE SAME SURVEY, still open and all in lane: the rollout's per-step batch-1 input tensor is rebuilt every step at a loop-invariant shape, about 17 percent of rollout allocations, and reuse is legal there only because the rollout context is untaped; the DQN target-build nest copies a same-shaped tensor element by element where one memmove would do, and samples a batch by struct copy where indices would do; and the PPO update rebuilds an epoch-invariant states tensor on each of eight epochs. The survey also confirmed, with reasoning I verified, that batching the actor forward across rollout steps is genuinely illegal: the next observation depends on the action sampled from the current forward, so the dependency chain is serial and the trajectory parity test would fail at the first divergence.

## R-01KYRF5DXREA49H2B6C0X73SE2 Decoding-path survey: the beam-search sorts are the lever, and they exposed a structural recall gap in perfscan's own sort checks
kind: research
state: draft
created: 2026-07-30

Delegated survey of the logit-processing and search files in nlp: samplers, warpers, penalties, beam and diverse-beam search, constrained decoding. Its most valuable output was not a candidate but a tool defect, which is now fixed.

THE TOOL DEFECT, VERIFIED AND FIXED. Both beam search and diverse beam search sort every candidate (beams times vocabulary) and keep only the top few, and PS6001 and PS6013 fire on NEITHER. Confirmed directly: both report zero hits across all of nlp. The gap is structural rather than a threshold — PS6013 requires a counted loop that indexes the sorted slice, PS6001 requires a consumer that breaks on a threshold, and in both sites the consumer is a reslice, so there is no loop for either to match. Shipped as PS6022 with four true positives and zero false positives tree-wide. The lesson to carry: silence from PS6001 and PS6013 is not evidence that a sort is well-sized.

MEASURABLE CANDIDATES, still open, all in lane. Beam search sorts every candidate but its consumer walks only until the frontier fills, so it never reads past rank width-plus-live-count except in the terminal step; quickselect on the existing total-order comparator followed by sorting the kept prefix is bit-identical for non-terminal steps, and the survey's argument that the terminal step's return is also unchanged rests on all candidates in a step sharing the same length penalty, which I have not verified. At benchmark geometry that is nineteen sorts of 16384 candidates for a consumer that reads eight. Diverse beam search is the same shape, eighty sorts of 4096 keeping two, and additionally recomputes its augmented sort key twice per comparison — about 98 thousand closure calls per group-step where 4096 would do; that one shrinks once the selection lands, so sequence it second. The log-softmax row allocates a vocabulary-sized slice per beam per step, roughly 2.4 megabytes per call in both searches, and the returned slice is consumed immediately and never retained. Both candidate buffers are allocated per step with a capacity hint 256 times too small, so each grows through about eight doublings.

CANDIDATES BLOCKED ON A MISSING BENCHMARK: the guided-decoding sampler copies a vocabulary-sized slice per token and its mask loop reaches a memoized transition table through a non-inlinable double dispatch per token; the watermark wrapper copies a vocabulary-sized slice per token, while the existing benchmark measures only the public allocating API and so cannot show the change. Contrastive search recomputes both vector norms for every candidate-context pair where two thirds of that work is loop-invariant.

WHAT THE SURVEY CONFIRMED IS ALREADY DONE, which is the useful negative result: the sampler file is fully pooled, its selection paths already use quickselect and a radix sort above a cutoff, and there is no remaining per-element tensor dispatch anywhere in the lane. The prior negative verdict on the per-token row-logits allocation was found and respected rather than re-derived.

REJECTED WITH REASONS. Parallelizing anything in this lane is illegal: every vocabulary-sized loop either carries a floating-point accumulation — softmax normalizer, cumulative probability scan, entropy, the standard-deviation sums — or is the multinomial scan that consumes the RNG. Replacing a per-lane divide with a reciprocal multiply in Mirostat and XTC was rejected as a bit-changing edit to a feedback-coupled sampler, where a one-ulp move can flip the draw boundary and every later token diverges; that needs an explicit tolerance decision, not a bit-identity label. Two sort-to-selection sites in the constrained-decoding construction path are cold behind per-state memoization. Also recorded: the Mirostat benchmark's documentation is stale — after a probability pre-filter landed, its geometry no longer reaches the descending sort it claims to exercise, leaving that path uncovered.

## R-01KYRGM1ABEHD9C62M3Y98S1B7 Beam-search selection: the bound is provable, but plain quickselect lost to pdqsort on this package's own benchmark fixture until it became an introselect
kind: research
state: draft
created: 2026-07-30

Implements the largest measurable candidate from the decoding survey. Three findings worth more than the change itself.

WHAT SHIPPED. BeamSearch sorted every candidate, beams times vocabulary, then walked only until the frontier filled - a sort of 16384 to consume at most 16 at benchmark geometry. It now selects a bounded prefix first. Allocations fell 95.9 percent, bytes 10.5 percent, the realistic arm 20.5 percent, the cheap arm 2.1 percent.

THE BOUND IS PROVABLE RATHER THAN TUNED, which is what makes the truncation bit-identical instead of approximate. The walk stops once the frontier holds width, so it consumes width survivors plus the completions it passes; in a non-terminal step a candidate completes only through the end-of-sequence token and the expansion gives each parent exactly one such candidate, so completions are at most the live count. The terminal step needs a separate argument because there every candidate completes and the frontier never fills: sequence length is uniform within a step by induction, so the length penalty is a single positive constant and final-score order equals raw-score order, hence a dropped candidate is worse than all retained ones and cannot reach the returned top width.

PLAIN QUICKSELECT LOST, AND THE REASON GENERALIZES. The first version regressed the cheap benchmark arm 17 percent while improving the realistic arm 20 percent. Isolating the selection from the accompanying edits - all of them together with selection disabled reproduce the baseline exactly - pinned it to the selection, and a two-shape micro-benchmark explained it. On varied candidates the selection beats pdqsort 61 microseconds to 1236. On the cheap fixture, where the model returns identical logits for every prefix so the array is eight near-identical copies of one smooth curve, median-of-three picks poor pivots repeatedly and the same selection takes 1442 microseconds against pdqsort's 1170 - a 24-fold self-slowdown, not a small constant. Bounding partitions at twice the log of the length and finishing with a sort caps the bad case at 1201 while leaving the good case at 61. Cast as a rule: a selection replacing a sort must be an introselect, and must be measured on the real input distribution, because uniform random test data would never have surfaced this.

A MEASUREMENT TRAP, the second of this kind in two iterations. Restoring the base file while deleting the new helper left a test referencing the missing symbol, so the base arm failed to COMPILE and produced zero samples; benchstat printed a single-column report of the new arm alone, with no error, which reads exactly like a completed comparison. The existing rule about verifying base-arm source text passes in this case - only a build check catches it - so that is now its own rule.

THE GATE. Comparison against the pre-selection algorithm transcribed verbatim across 1728 configurations sweeping vocabulary, width, length cap, end-of-sequence on and off, and three length-penalty values, with tie-heavy arms whose logits contain long runs of exactly equal values, since ties are precisely where a selection may legitimately keep a different set. Tokens and score bit patterns compared exactly. Four mutations turn it red: dropping the completions allowance, an off-by-one in the bound, removing the comparator's tie-break, and corrupting the partition.

STILL OPEN in the same lane: diverse beam search has the same shape twice over, at eighty sorts of 4096 keeping two, and additionally recomputes its augmented sort key twice per comparison; the log-softmax row allocates a vocabulary-sized slice per beam per step in both searches; and both candidate buffers are allocated per step with a capacity hint 256 times too small.

## R-01KYRGN5B7FKDT2742V77V70QZ KV-cache and speculative-decoding surveys: the quadratic-growth conversion is complete except one driver, verification already batches everywhere, and four analyzer false positives are diagnosed
kind: research
state: draft
created: 2026-07-30

Two delegated surveys, compacted. Their most valuable output is negative: the two architectural defects one would expect to find are already fixed everywhere they matter.

QUADRATIC CACHE GROWTH IS FIXED EVERYWHERE BUT ONE DRIVER. All forty-plus per-architecture append sites, including every quantized twin, are on the amortized row-buffer path. The two remaining concat call sites are an exported one-shot API, correctly left allocating, and StreamCache's step function - the single decode driver never converted. There it appends by rebuilding the whole cache and then immediately copies the result again to bound it to sinks plus a window, so two allocations and two full copies per layer per token, and genuinely quadratic during warm-up before the bound binds. The retained row set is exactly the concatenation minus one row, so a fixed backing buffer plus a single overlapping copy replaces both. Unmeasured: nothing benchmarks the streaming driver at all.

SPECULATIVE VERIFICATION ALREADY BATCHES, in all nine drivers - each scores every drafted position in one forward. There is no batch-one-in-a-loop to collapse, which is the shape one would most expect to find in this family. The drafting loops are all serial by data dependence, and three of them consume the sampler's random stream per drafted token, so reordering is illegal there and merely hoisting allocations is the only safe move. One real batch-one shape exists in the Medusa heads, but it would need a derived weight cache invalidated on every optimizer step and saves only a couple of dispatches; recorded as do-not-implement.

MEASURABILITY IS THE BINDING CONSTRAINT. Of nine speculative files, exactly one has a covering benchmark, and it runs at a vocabulary of seven, where a vocabulary-sized allocation is 56 bytes - it can measure allocation counts credibly and wall time not at all. The highest-leverage item found is not in a decode loop: two model constructors fill weights through per-element index-unravel plus variadic set, which is two heap allocations per weight element, over a million for realistic dimensions. That is trivially isolatable in a constructor micro-benchmark and needs no new fixture.

FOUR ANALYZER FALSE POSITIVES, diagnosed precisely, which is the tool feedback this campaign needs. A per-element write flagged inside what is already a guarded fallback arm below a bulk-copy fast path. An integer-keyed map flagged where the keys are token identifiers drawn sparsely from a large vocabulary, so a dense slice would be vocabulary-sized per node. A dispatch-heavy loop flagged as a fusable scalar recurrence when it is in fact the transformer block stack, carrying no scalar state and dispatching large matmuls - the most-fired check in the package, so a large share of its hits are likely this. And a movement-cluster hit inside a training-only loss, where the recommended fix gates on an untaped context that this function never has; that last one matches a rejection I already recorded for a different site, so it is now the second instance and the suppression is worth building.

TWO CHECKS CONFIRMED CLEAN: the fan-out-without-a-seam check fires zero times across this package, and both surveys looked for the shape deliberately. The sort-then-truncate check fired only on the beam files, which is correct - the eviction code already uses the selection it prescribes.

ALSO RECORDED: a quantized KV store dequantizes row by row where the block format allows one call over the whole range, an eviction path allocates a full replacement cache per layer where an in-place forward compaction would do because every keep-list it produces is ascending, and a per-layer offload context is rebuilt per token from values fixed for the whole generation. All three are allocation wins with no in-package caller hot enough to measure today.

## R-01KYRHKDQ0FMTARMY93KY01TY0 Eigensolver and SVD survey: 98 percent of the sweep is legitimately off limits, so the wins were on the accessor and allocator axes
kind: research
state: draft
created: 2026-07-30

Delegated survey of the eigensolver, singular-value decomposition and derived-quantity files, with the already-optimized solve paths read for their idiom but excluded from proposals. Its central result is a confirmation rather than a discovery, and that is what made it useful.

WHY THE OBVIOUS TARGET IS CLOSED. A prior pass profiled the decomposition at roughly 35 milliseconds with the three-way norm accumulation at 74 percent and the column rotation at 24 percent. Both are unavailable: the accumulation is three floating-point reductions over the axis one would split, and the pair loop carries a true dependence because rotating a column pair mutates columns that later pairs read. A parallel Jacobi ordering exists but is a different algorithm, not a rounding difference. The maintained-norm identity that would collapse the dominant loop from three accumulators to one is likewise not bit-identical and compounds across roughly twelve thousand rotations. So 98 percent of the hot function was correctly declined before this survey started, and the honest scope was the remaining two percent plus the allocation axis nobody had examined.

SHIPPED. The two induced-norm functions read every element through the tensor accessor, which the survey verified is not inlinable - a real call plus a storage type switch against a three-cycle add chain. The file already defined the flat-storage helper for exactly this and had simply not applied it to the norms. Measured 3.1 times faster on the covering benchmark. Separately, the decomposition built its working column copy and its accumulator as separately allocated rows; both became views over one backing array, cutting pseudoinverse allocations 64.6 percent at the larger geometry with bytes unchanged.

DELIBERATELY NOT DONE. The one-norm was left alone: its loop is column-outer so the flat form would still stride, and the bit-identical relayout costs an allocation that no benchmark can justify. The transposed-flat relayout of the accumulator is bit-identical and correct but both benchmark geometries sit below cache, where the prior campaign measured the identical transformation at only 1.05x; it needs a larger size to be visible. The pseudoinverse accumulation loop nests its reduction index outermost, which is a real defect, but both existing geometries fit in first-level cache so it is unmeasurable there - it needs a tall benchmark, and the repo's own two-size rule says a layout change cannot be validated when every arm is cache-resident.

A CONTROL BENCHMARK WAS INVALIDATED, which is worth its own rule. The norm benchmark exists as the CONTROL for pseudoinverse comparisons, chosen because it touches the same tensors without entering the accumulation loop. Making it 3.1 times faster moved that baseline, so any later comparison must re-establish its floor rather than reuse the recorded one.

TWO ANALYZER DEFECTS DIAGNOSED PRECISELY. A radix-sort suggestion fires on a sort whose length is a matrix dimension - eight to fifty elements - where a radix pass loses badly; the check would be fixed by suppressing when the sorted slice length is provably derived from a shape. And a flatten suggestion on a row-of-slices matrix proposes the row-major layout when the loop reads DOWN a column, so the suggested rewrite delivers the allocation win and none of the locality win while reading as though it delivered both; it should emit the transposed layout when the inner loop variable sits in the row position. A third hit correctly identifies that the pseudoinverse's fallback arms are unreachable dead code, which is a test-coverage finding rather than a performance one.

COVERAGE GAP WORTH RECORDING: the eigensolver entry point has no benchmark anywhere in the tree. Two benchmarks whose names suggest otherwise both bypass it - one calls the shared kernel directly, the other exercises a separate reimplementation in the backend.

## R-01KYRHMQ0NF6EABEA9JH9C3V10 Data-prep and adapter survey: seven of nine files have no benchmark, and the one measurable candidate is a document mask fill
kind: research
state: draft
created: 2026-07-30

Delegated survey of the sequence-packing, corruption, pooling and adapter-injection files. The dominant finding is coverage: seven of the nine files are reached by no benchmark at all, so most of what it found cannot be validated on this host and is out of bounds for implementation.

MEASURABLE NOW, one candidate. The document causal mask fills an n-by-n tensor with a branchy per-element store whose predicate compares two document ids. Rows sharing a document id produce identical masks, so one template per distinct id turns the fill into a memmove per row. Bit-identical - the template entry is zero exactly when the ids match, and the tail is unconditionally the masked value. The covering benchmark already exists at the right shape, 2048 by 2048 over eight documents. The honest ceiling is around two to two and a half times rather than the four the fill-loop delta suggests, because allocating the output tensor already pays one full zeroing pass the fix cannot remove.

HIGHEST INTRINSIC LEVERAGE, BUT UNMEASURED. Sequence packing does a first-fit scan that is quadratic in the item count over the block count, appends token by token into slices that start nil, and sorts through the reflection-based stable sort. All three are fixable without changing the output - a segment tree finds the same leftmost fitting block, the appends become one bulk copy per sequence, and the sort converts to the generic stable form with the index-versus-value flip that conversion requires. Nothing in the repository calls this above seven tokens, so it needs a benchmark before any of it can be justified.

A LEGAL, COMPUTE-BOUND PARALLELIZATION, which is rare in this campaign. The cosine reranking loop partitions cleanly over candidates: each candidate's dot and norm accumulate privately and each writes its own output slot, so splitting is bit-identical, and unlike the elementwise sweeps that returned only 1.10x this one is latency-bound on two independent multiply-add chains rather than bandwidth-bound. The honest cap is still modest because the sort over the results costs about as much as the scoring and is untouched.

REJECTED, with reasons worth keeping. The composition sampler draws one random number per token where a multinomial would need one per span - the single largest cost in the corruption path - but any reformulation changes the draw sequence and would require regenerating fixed-seed goldens, which is a maintainer decision rather than a mechanical optimization. The masking loop is strictly order-dependent for the same reason. The adapter constructors allocate a small map per block, which is a genuine shape but runs once per model attachment. A pooled-embedding mean cannot be split over its accumulation index, and the legal axis is bandwidth-bound.

ANALYZER FEEDBACK. Two hits are false positives by hotness, both inside constructors that run once per model. One radix-sort suggestion fails both of its own stated preconditions: the key is composite, and the score is a cosine similarity that is routinely negative, where the bit-pattern ordering a radix relies on inverts. One map-keyed hit is the opposite of a previously diagnosed case - here the keys ARE dense over a shifted interval, so the check was right about density and wrong only about hotness, which suggests the check could usefully distinguish dense-shifted-interval keys from arbitrary identifiers. One hit correctly reports that a three-armed copy helper has an untested fallback arm, with a sibling in the same package showing exactly the oracle test it lacks.

## R-01KYRJMVNMFFD9KMENJNF35663 Swin and gradient-boosting surveys: one large parallel candidate, and a coverage gap where the faster code arm is never benchmarked
kind: research
state: draft
created: 2026-07-30

Two delegated surveys, compacted. Both were unusually disciplined about magnitude, and both produced negative results worth more than their candidates.

SWIN. The window-attention loop over (window, head) pairs is 288 independent units per forward issuing 864 of the roughly 1015 backend dispatches, and every one runs on a single core because each unit's matrices are far below the backend's own parallel threshold. The parallel index is legal - the softmax maximum and denominator are computed inside one unit over its own score row, and the placement writes disjoint slots - and unlike the elementwise sweeps that returned only 1.10x this one is compute-bound at roughly 3.3 flops per byte, so the bandwidth calibration does not transfer. The blocker is that the fill buffers are currently shared mutable state and must become per-worker. Estimated 1.4 to 2.2 times overall but explicitly conditional on profiling the dispatch-versus-GEMM split first.

The survey also found that this file is the one caller that never migrated to the shared patchify helper the rest of the package uses, still paying a per-image staging buffer and a closure call per pixel; and that its geometry preamble - shift indices, partition indices, inverses, masks, index tensors - is rebuilt every block of every forward from values that depend only on geometry. Both are allocation-axis items at this benchmark size.

A REACHABILITY BLIND SPOT WORTH FIXING IN THE TOOL: three of the four remaining analyzer hits in that file sit in fallback arms that are dead whenever the fast path is taken, which is every arm of the covering benchmark. The analyzer has no model of a boolean guard that a fast path sets, so it reports code that never executes.

GRADIENT BOOSTING. The exact-versus-histogram benchmark pairing is genuine - the two arms really do diverge at the grower branch, so this is not the degenerate-regime defect found in the clustering benchmark earlier. But three code arms inside the histogram builder are dead at both benchmark sizes, and one of them is the FASTER serial inner form, which the file's own comment measures as 22 percent better than the arm that does run. The faster arm is unbenchmarked, so any change to it would be unvalidated.

The leading candidate is a layout flip: the bin-code array is row-major while every hot reader walks it down a column, so a feature-major pass touches every cache line of the whole array to consume two bytes from each. Bit-identical, zero-allocation, helps three sites. Honestly sized at single-digit percent by back-solving Amdahl on the recorded parallelization win. The second is a loop interchange in forest prediction, making one tree cache-resident across a chunk of rows instead of sweeping all hundred trees per row; bit-identical for both classifier votes and regressor sums, but explicitly capped at 1.15 to 1.4 times rather than the 2 times the working-set arithmetic suggests, because a walk touches only a dozen of a tree's nodes.

THE MOST USEFUL NEGATIVE: several allocation candidates here are real but dwarfed. One removes 900 of 352008 allocations, another 100 of them. The dominant term is the tree builder's index partitioning, which is another engineer's lane. Reporting those as wins would be misleading and the survey says so directly.

THREE ANALYZER FALSE POSITIVES DIAGNOSED. An allocation-in-parallel-body hit where the dispatch is once per worker and the enclosing call is a top-level API entry - which is the CORRECT per-worker seam, not the defect. The same check firing where the enclosing constructor runs once per fit ahead of fifty rounds. And an integer-keyed-map hit whose keys are arbitrary user labels rather than a dense range. Also confirmed: the fan-out-without-a-seam check correctly does NOT fire on this package's work-queue primitive, because the channel exemption built into it is doing its job.

## R-01KYRM0MCSEAS8Q2P7TTNYNNSE Pool survey: the Swin profile is entry churn in the backend pool, and every parallel threshold in the tree rests on an unmeasured dispatch cost
kind: research
state: draft
created: 2026-07-30

Delegated survey of the shared fan-out and kernel packages, prompted by a Swin profile showing four fifths of samples in pool synchronization. Its lead finding is a diagnosis rather than a candidate, and it is the most useful thing in it.

THE PROFILE IS ENTRY CHURN, NOT IDLE PARKING, AND IT IS THE WRONG POOL. The vision path does not use the shared helper at all; it reaches the compute backend's own pool. The discriminator is the wake counter rather than the wait counter: nothing ever parks inside a condition-variable signal, so a double-digit share there is time spent waking - on the order of a hundred thousand wake operations inside a seven-hundred-millisecond window. A pool entered a few times with workers parked throughout would show that at nearly zero. The backend pool has a dense regime with spin-before-park and caller help, gated on total work below a size the Swin matrices exceed, so Swin runs the sparse regime, which parks directly and pays a full wake cycle per worker per barrier. The backend's own comment records the same signature measured on a decode benchmark at 71 percent. This also explains why my own fan-out returned 1.11x rather than the predicted 3.3x - the wall clock is not where the one-core profile said it was.

THE SHARED HELPER HAS NO BENCHMARK OF ITS OWN DISPATCH COST. Eleven separate parallel thresholds across the tree are implicit claims about that cost, and nothing measures it. The dominant constant is described in two places as measured and no artifact of that measurement survives. The survey's estimate, reasoned from the channel and wake operations per fan-out, puts the break-even well above the constant every caller uses, meaning several call sites may be admitting the parallel path below its own crossover. That estimate is unverified; the actionable item is the microbenchmark, run both warm and cold, because the two should differ several-fold.

THREE DISPATCH MECHANISMS COEXIST: the shared pool, the backend pool, and one file spawning raw goroutines per chunk - twenty-two permanent workers on twelve processors. Nesting does not deadlock, because both pools submit non-blockingly and fall back to inline, and that safety rests entirely on the non-blocking submission, which is worth knowing before anyone makes it blocking.

CANDIDATES. The chunk-indexed variant of the shared helper allocates one closure per chunk, roughly a dozen per call, which is exactly the shape that function exists to eliminate for its callers; the fix is to carry the index in the task struct. A scalar scan kernel walks three arrays at a stride equal to the channel count, costing eight-fold read amplification, and its correct loop order already exists in a sibling - but the loop calls an exponential per element, which swamps the traffic, so the survey caps it at single digits rather than the ratio the analyzer quotes. A butterfly transform lacks the bounds-check hoist the rest of its file documents as worth 1.45 times, with the caveat that this loop has a store-back dependency the measured one did not.

WORTH KNOWING: the vectorized kernels for the other architecture cannot be measured on this host at all, so their analyzer hits are unactionable here; and on this architecture several benchmark pairs named for a vectorized and a scalar variant call the same function, so they are duplicates rather than comparisons.

## R-01KYRMHWX1EWDB7WF6JTV2WYM2 The shared fan-out helper is 15 percent slower than serial at the threshold eight call sites use, and cold dispatch costs no more than warm
kind: research
state: draft
created: 2026-07-30

Builds the dispatch-cost benchmark the previous survey identified as missing, and reports what it says. Both results contradict a standing assumption.

THE THRESHOLD IN WIDEST USE IS BELOW ITS OWN CROSSOVER. Running the identical elementwise body serially and through the shared helper: at 1024 elements the parallel path is 0.13 times as fast, at 16384 it is 0.74, at 32768 - the constant used at eight or more call sites - it is 0.85, and it first wins at 65536 with 1.20. So the crossover sits between two-to-the-fifteenth and two-to-the-sixteenth, roughly double the constant in use, and at exactly that constant the parallel path LOSES fifteen percent.

THE CAVEAT THAT DECIDES WHICH SITES ARE IMPLICATED. The body is one multiply-add per element, about twenty-four bytes of traffic per element, which is the bandwidth-bound regime where parallelism gains least. Sites whose threshold counts raw ELEMENTS over a comparably cheap body - the rotation and solve loops in the linear-algebra package - are directly implicated. Sites that multiply by a per-element work factor, or whose body calls a transcendental, may be fine at the same numeric constant. That distinction is why no threshold was changed here: each site needs its own measurement, and the deliverable is the instrument.

COLD DISPATCH IS NOT MEASURABLY MORE EXPENSIVE THAN WARM, which refutes the expectation that motivated measuring them separately. A two-hundred-microsecond idle gap before each fan-out leaves the result statistically indistinguishable at three sizes spanning the crossover. One threshold suffices; the warm-versus-cold split can be dropped from future reasoning about this pool.

SHIPPED ALONGSIDE: the chunk-indexed variant no longer allocates a closure per chunk - thirteen allocations and 312 bytes per call against the plain variant's two and forty-eight. That is precisely the per-call allocation the function exists to spare its callers, and its own documentation records a thirty-one-fold memory regression from that shape in the boosting grower. The index now rides in the task struct; bit-identical, and measured at no time cost.

A FOURTH MEASUREMENT TRAP, and the reason that last claim needed two measurements. The paired benchmark first showed the fixed variant eighteen percent SLOWER, which read as a real dispatch regression. It was the two arms running back to back over a shared slice, so the second inherited a different cache and scheduler state; independent arms per size put the two within 1.00 times from two-to-the-fourteenth to two-to-the-twentieth. The paired benchmark now documents that it is for allocation counts only. Cast as a rule, alongside one requiring a threshold constant to cite the benchmark that located its crossover.

## R-01KYRMYADSF83BGTR0EJT9VF0S Eigensolver bounds checks were worth 1.18x, ten times the estimate, which recalibrates the whole issue-width class; and the analyzer's stride hits there are correct but non-actionable
kind: research
state: draft
created: 2026-07-30

Implements the one bit-identical candidate from the internal eigensolver survey and reports a result that should change how this class is estimated.

WHAT SHIPPED. Six bounds checks per inner iteration survived across the Jacobi rotation's three loops. Four are now gone, measured at 1.18 times at n=64 and 1.14 at n=128, bit-identical, with the existing tolerance-zero oracle green and mutation-verified.

THE ESTIMATE WAS OFF BY TEN TIMES, AND THAT IS THE USEFUL PART. The survey put this at 1.03 to 1.10 times if the loop was issue-bound and 1.00 otherwise, and explicitly framed the measurement as the calibration deciding whether the remaining issue-width candidates - loop fusion with a four-element peel, and unrolling - were worth attempting. Two of thirteen uops per iteration turned out to be worth fifteen percent, so the loop has substantial issue-width headroom and those candidates are now worth trying rather than dismissing. The general lesson: an issue-width estimate derived from counting uops against a core's nominal width understates badly when the loop is short and store-heavy.

A NON-OBVIOUS PRECONDITION, now cast as a rule. The eigenvector loop already had the row-slice hoist and still carried both of its checks. Hoisting a slice is not sufficient - ranging over a separate element count leaves the prove pass with no relation between that count and the slice length. Clamping the second slice to the first and ranging over it is what discharges them. Anyone applying this shape should confirm with the compiler's bounds-check debug output rather than assuming the hoist worked.

DELIBERATELY NOT TAKEN. Folding the two symmetric rotation loops into a triangle-only form is the largest candidate in the package at an estimated 1.30 to 1.45 times, and it is REASSOCIATING: the lower triangle feeds back into the column loop's reads, and the two triangles differ by up to one ulp after every rotation. Taking it would retire the bit-identity oracle and put every downstream ulp-sensitive golden in play. That is a decision about the numerics contract, not an optimization.

THE ANALYZER HITS THERE ARE CORRECT AND NON-ACTIONABLE, which is a distinct verdict from false positive and worth recording as such. The strided column walk is real, but the suggested interchange is impossible - the enclosing loops ARE the rotation sequence, so there is no loop to interchange with - and the suggested blocking of four adjacent columns is illegal, because each rotation's angle is computed from entries the previous one just wrote, so blocking reorders the rotations into a different algorithm. Independently, the cost has already been measured away: the package scales at 8.35 times from n=64 to n=128 against a cubic prediction of 8.00, leaving about four percent of cache penalty in the entire larger run, and consecutive columns already share cache lines and are visited back to back.

TWO OTHER RESULTS FROM THE SURVEY WORTH KEEPING. It independently rediscovered and confirmed a previously recorded rejection about symmetry not being exact, deriving the same one-ulp asymmetry between the two triangles from the expression trees. And it killed its own leading hypothesis with arithmetic: an absolute convergence threshold suggested all one hundred sweeps might be running, which the recorded runtime refutes, since that would require a quarter of a cycle per iteration against two stores per iteration. The surviving concern from that line is only that the threshold is scale-dependent, which is a robustness matter with no benchmark to show it.

## R-01KYRNABFXFYY8PZXSBTJBQY40 Bounds-check removal generalizes but its payoff does not: 15 percent in a rotation kernel, 3.3 percent in an early-exit kernel, and 11565 live checks tree-wide are not a usable signal
kind: research
state: draft
created: 2026-07-30

Follows the eigensolver result by asking where else the same transformation pays, and answers with a calibration rather than a scan rule.

THE TREE-WIDE CENSUS IS NOT ACTIONABLE. Compiling everything with the bounds-check debug flag reports 11565 live checks, concentrated in the neural-network, language and autograd packages. That number cannot be ranked: most sit in cold code or on paths executed once. The eigensolver's mattered because they were inside an O(n-cubed) inner loop, and nothing in the census expresses that. This is the same wall the hotness rule already records - a count of syntactic occurrences says nothing about which ones run.

SO THE INSTRUMENT IS A PROFILE, NOT A SCAN. The workflow that worked: profile a covered benchmark, take the top flat function, then check that function for live per-iteration checks. Applied to the hottest kernel in this lane it found three ball-tree distance functions all ranging over one row while indexing the other, which leaves the prove pass no length relation - the identical shape the eigensolver had, in a loop measured at 25 percent flat.

SHIPPED: clamping the second row to the first discharges every per-iteration check in all three kernels, leaving one slice-bounds check per call. Measured 3.3 percent on the nearest-neighbour predict benchmark; the clustering arms did not move.

THE PAYOFF DIFFERS BY FOUR TIMES BETWEEN THE TWO SITES, AND THAT IS THE FINDING. Fifteen percent in the rotation kernel, 3.3 in the distance kernel. The rotation loop is pure arithmetic, so the checks were two of roughly thirteen uops and issue width was the binding constraint. The distance loops carry a data-dependent early-exit compare that mispredicts on the far points dominating a leaf test, so the branch is what the loop waits on and the check is nearly free to begin with. Counting uops against issue width predicts the first correctly and overshoots the second fourfold. Cast as a rule: discount sharply when the loop contains an unpredictable branch.

WHY NO SCAN RULE WAS ADDED. The compiler already detects this shape exactly, and better than an AST approximation could - it reports the checks that actually survive, after inlining and after prove. An AST rule matching range-over-a-count-while-indexing-another-slice would flag thousands of sites the compiler has already discharged, and would still not rank them. The honest division of labor is: the compiler finds the checks, a profile finds the hot ones, and the analyzer stays out of it. That is the second time this campaign has concluded that a real and generalizable optimization should NOT become a scan rule, for the same underlying reason.

STILL OPEN, now with proven headroom: the eigensolver's loop fusion with a four-element peel, and its unrolling candidate. The 15 percent result established that its inner loop is issue-bound, which is exactly the condition those two need and which was previously unknown.

## R-01KYRNP2D9EN2V7JVB9C58X9T4 Eigensolver fusion landed for a further 3.5 and 7.2 percent, and the peel the legality argument demanded is confirmed by mutation
kind: research
state: draft
created: 2026-07-30

Completes the internal eigensolver line. Cumulative 1.21 times at n=64 and 1.22 at n=128 across two bit-identical changes, both gated by a tolerance-zero oracle that already existed.

THE SEQUENCING IS THE LESSON. The survey estimated bounds-check removal at 1.03 to 1.10 times and explicitly framed it as the calibration that would decide whether the fusion and unrolling candidates were worth attempting at all. It measured 1.18, which established the inner loop is issue-bound; the fusion that answer unlocked then added a further 3.5 percent at n=64 and 7.2 at n=128. Had the cheap candidate been dismissed on the strength of its small estimate, the expensive one would have stayed closed too. Cast as a rule.

WHICH LOOPS FUSED AND WHY NOT THE THIRD. The column rotation of the matrix and the rotation of the eigenvector accumulator are disjoint by construction, so they fuse with no peel and no skip and their dependency chains issue together. The row rotation is left separate because at the two pivot indices it reads the four cells the column rotation has just written, so fusing it requires a four-element peel of both loops.

THAT CAVEAT IS EMPIRICALLY CONFIRMED, not just argued. Fusing the row loop naively, without the peel, turns the bit-identity oracle red. The legality analysis and the test agree, which is worth recording because the analysis was subtle enough to be worth doubting - the overlap is exactly four cells out of an n-by-n matrix, and a naive reading would call the loops independent.

THE SIZE DEPENDENCE CAME OUT IN THE PREDICTED DIRECTION. Fusion helps roughly twice as much at n=128 as at n=64. At 256 kilobytes the working set is out of first-level cache, so overlapping a strided stream with a contiguous one buys more than when everything is already resident. The bounds-check removal, by contrast, was nearly size-independent, which is the correct signature for a pure instruction-count change.

STILL OPEN, and now the only remaining candidate in the package: unrolling the fused loop by four, bit-identical since the iterations write distinct destinations and carry no accumulation. Also still open and still declined: the triangle-only symmetric form, estimated at 1.30 to 1.45 times but reassociating, which would retire the oracle both of these changes were validated against.

## R-01KYRPFK1NEG5A8EAECNVQA4T5 An audit of the most-fired scan check found zero of its 110 hits matched its own shape, and the counted fix prunes 84 of them
kind: research
state: draft
created: 2026-07-30

Closes the precision question two earlier surveys had raised in passing, with the arithmetic they lacked.

THE RESULT. Every one of the check's 110 hits was classified. None was the sequential recurrence its message described. Fifty-seven were transformer layer stacks, thirty-five more were per-head, per-window or per-expert fan-outs, twelve were movement-only preparation loops. The strict false-positive rate - the message's premise being factually false at the site - was 109 of 110.

WORSE THAN THE RATE: the genuine class exists, six loops in this tree, every one carrying explicit state across a sequence-length range, and all six were already suppressed by a different guard. So the check was reporting 110 sites and had never once reported one of the six it was built for. A high hit count read as high yield and meant the opposite.

THE FIX IS ONE NEGATIVE CONDITION, counted before it was written and verified after: skip the loop when its trip count comes from a FIELD. Layer, head and expert collections are architecture counts on the order of tens; a sequence length arrives as a local or a parameter. Measured 110 hits down to 26, all six genuine sites kept, the three existing fixtures unchanged, layer stacks down from 57 sites to 1.

FOUR RICHER PREDICATES WERE COUNTED AND REJECTED, and the arithmetic is the deliverable rather than the idea. Loop-carried state fires on every layer stack too, because the residual IS carried state - it is what the two shapes share, not what separates them. Requiring the carried value to feed two or more dispatches loses the canonical single-chain recurrence, dropping recall to four of six. Filtering elementwise against matmul ops has recall ZERO, because twenty-two layer stacks show no visible matmul - theirs sit behind method calls an AST walker cannot resolve - while all six real recurrences do show one. Matching a literal row-slice attribute also has recall zero.

THE MESSAGE OVERCLAIMED and is corrected. It asserted a per-sequence dispatch cost and prescribed a plain-Go recurrence, both true of zero hits. It now states the condition and tells the reader the check cannot see the trip count.

TWO ADJACENT DEFECTS, recorded not fixed. The fused-path guard matches a single identifier, so it misses every fused path spelled otherwise - thirty-six of the hit functions already have one, and in one vision file this rule's own documented success case was being re-reported as an opportunity. And the check reports the outermost qualifying loop per nest but not siblings, so one function can still yield several hits.

THE PROCESS POINT, cast as a rule: a check that accumulates dozens of unacted-on hits should have every hit classified before it is treated as a work queue. This one had two prior surveys note in passing that its premise was wrong for most matches; neither counted, so nothing changed for months.

## R-01KYRPSF2YFS9VPA1MVR6Q4GF6 The fused-path guard generalization was worth one hit, not the thirty-six the pre-filter count implied, so it was declined
kind: research
state: draft
created: 2026-07-30

Closes the second of two adjacent defects the scan-check audit surfaced, with a negative result that only appeared once the counting was redone in the right order.

THE DEFECT AS REPORTED. The check skips a function that already carries a fused fast path, but it tests for a single identifier, so it misses every fused path spelled otherwise. The audit put that at thirty-six of the original one hundred ten hit functions - a substantial-looking second win on top of the trip-count filter.

THE RECOUNT CHANGED THE ANSWER. After the trip-count filter cut the check from 110 to 26, I classified all twenty-six surviving enclosing functions. Only four carry any fused-path signal, and only ONE carries the tight one - an inference-only arm gated on an absent recorder, which is semantically exactly what the guard means. The other three show merely a raw-storage access, a signal one hundred ninety-five files in this tree contain, far too loose to suppress on without risking genuine recurrences.

So the two filters overlap almost completely, and generalizing the guard buys a single hit at real recall cost. One instance is not a rule; the same arithmetic killed an earlier candidate built on two measured instances. Cast as a rule: recount a second precision fix against the SURVIVORS, not the original population, because the pre-filter figure describes a population the first filter has already removed.

WHAT WAS DONE INSTEAD. The one site is genuinely misleading rather than merely redundant - the check was pointing at a dispatch fallback and recommending it be fused, when the fused version is the code immediately above it, and that same site is recorded elsewhere as one of this rule's own success cases. A suppression directive at the site carries the reason where a reader sees it, and the unused-directive check added earlier now catches it if a later edit drifts it away from its target. Verified: the check drops from 26 to 25, and the unused-directive check reports the new directive as live.

ALSO CHECKED, because it would have been the higher-value finding if true: whether the single-identifier guard pattern repeats elsewhere in the analyzer. It does not - exactly one check uses it. That is a clean negative and closes the question.

## R-01KYRQZQ2VEWYS66EC7YAAHXYZ The largest scan check audited SOUND at zero false positives, and the contrast with the one that failed is a fact-versus-cost-model distinction
kind: research
state: draft
created: 2026-07-30

Second hit-by-hit audit of a high-count scan check, run with the method that worked on the first. The result is the opposite and the reason why is the durable part.

ZERO FALSE POSITIVES OF 198, under both a strict and a lenient reading. Every clause of the message holds at every site, and the load-bearing one is compiler-verified rather than argued: all 198 lines emit an escapes-to-heap diagnostic. An independent recount of all 536 candidate calls in the two packages reproduced the check's output exactly, hit for hit, with zero misses and zero extras. The named sibling exists at every site and transfers argument for argument. One candidate false-positive class - training-only loss code where the pooled sibling would defer to the variadic form under a tape - collapsed on inspection, because eleven of twelve call sites pass an untaped context.

WHY THIS ONE IS SOUND AND THE OTHER WAS NOT. The failed check inferred a COST MODEL - a per-sequence dispatch overhead - from a loop whose trip count it could not see. This one asserts a FACT, that a slice is constructed at this line, plus a SIGNATURE TRANSFER, that a named sibling accepts these arguments. Both are fully decidable from the syntax tree. That is the line worth drawing when judging any future check: a claim about what the code IS survives static analysis, a claim about what it COSTS does not.

NO PREDICATE CHANGE. Three candidate filters were counted against the full hit list and all had precision zero on their removals, including transplanting the fix that saved the other check - it would have discarded 190 genuine allocations including every benchmark-reached one, for recall of four percent.

THE ACTUAL GAP WAS IN THE HELPER SET, NOT THE CHECK. It reports a variadic call only when a non-variadic sibling of that arity exists, and no four-input sibling existed, so five masked-attention allocations were structurally invisible. Declaring one closed recall from 198 of 203 to all 203 with no analyzer change. Cast as a rule.

SHIPPED: the new sibling plus 63 conversions, taking the check from 203 hits to 148. Measured six percent fewer allocations on both benchmarked decode paths, about sixteen and twenty thousand objects per operation, with no measurable latency change - which matches the check's own modest claim of one allocation per call and the campaign's earlier recorded range for the same transformation.

PRIORITIZATION, NOT PRECISION, IS THIS QUEUE'S WEAKNESS: 148 of 198 sites are on no benchmarked path, and each is worth a single small allocation. The wins concentrate entirely in the fifty reachable ones. Working it in benchmark-reachability order is the whole optimization.

THE GUARD BLIND SPOT DOES NOT REPEAT: the single-identifier helper is called from exactly one place in the analyzer. One sibling check ORs three broader arms and has only a narrow residual gap; a third has the same shape but zero hits, so it is latent rather than live.

## R-01KYRRXFDPE089QEV3FR5FYKD6 All 59 scan checks classified: the analyzer is broadly sound, and its whole liability is six messages that oversell borrowed numbers
kind: research
state: draft
created: 2026-07-30

Triage of every registry entry against one question - is the check's claim decidable from the syntax tree, or does it depend on how often the code runs, how big its data is, or what a kernel costs. The hypothesis came from two hit-by-hit audits with opposite outcomes and it held.

THE SHAPE OF THE ANALYZER. Fifty-nine entries, not the forty-four I briefed - that was the count with non-zero hits, and the agent corrected me. Thirty-eight are fact-class and carry 752 of 1288 findings; nine are cost-model, twelve mixed. Every one of the five largest checks by volume is fact-class, and together they are forty-two percent of all output. The analyzer is sound; the problem is narrower than expected.

THE LIABILITY IS SIX MESSAGES, 290 HITS. They assert a magnitude measured somewhere else as though it applied where the check fired. That is precisely what made the check audited earlier useless for months - a reader trusts the number, acts, finds nothing, and stops trusting the tool. Five checks already handle this correctly and one of them publishes its own counterexample, so the standard exists in-tree; six were below it. Fixed by attribution alone: no predicate touched, hit counts identical before and after.

THE MOST IMPORTANT SINGLE FINDING is not about performance. One check suggests replacing a loop-invariant divide with a reciprocal multiply, and with no type information it cannot tell float operands from integer ones. On integers the suggested rewrite evaluates to zero - a wrong-value bug, not a missed win. That now leads its message, and it is the one check whose next step should be a real audit, because its wrong answer produces incorrect code rather than wasted effort.

ANOTHER worth keeping: one check had no evidence and no hedge at all, the terse-est message in the analyzer attached to its third-largest cost-model count, and it prescribed a four-way unroll - the remainder-path shape another check in the same registry exists to warn about. Its message now names two checks that decide a site, one of which is answerable statically from the compiler's assembly output without any benchmark.

TWO CONCRETE DEFECTS, one mine. A check hard-coded its own identifier in its message while the reporter appends it, so every finding printed the ID twice; verified the class does not repeat. And a message credited a filter by a name I had invented in an earlier commit for a function that shipped under a different one - corrected to describe what it does.

WHAT WAS DELIBERATELY NOT DONE. No predicate changed. The triage floated one narrowing and could not count it, and the standing rule forbids proposing an uncounted predicate. Also recorded: the global summary footer already tells every reader that hits are candidates and hotness must be measured, which is why the cost-model checks are a message-honesty problem rather than a soundness one.

## R-01KYRSFK8RE02S269VSRPXR0KJ A scan check was recommending a rewrite that evaluates to zero at ten of its eighty-nine sites; three are now provably suppressed and seven need types
kind: research
state: draft
created: 2026-07-30

Audits the one check the whole-registry triage flagged as highest priority, and it was flagged for correctness rather than performance.

THE HAZARD. The check recommends replacing a loop-invariant divide with a reciprocal multiply. With no type information it cannot tell float operands from integer, and on integers the suggested reciprocal evaluates to ZERO — following the advice silently zeroes the result. That is a wrong-value bug, not a missed win, which is why it outranked the larger cost-model checks.

AUDITED ALL EIGHTY-NINE. Ten are genuine integer divides: a loop bound over a length divided by a stride; a group index feeding a scale table in two quantization dequantizers; four attention sites where a head count divided by a key-value count produces a head index; and one each in a compute backend, a weight loader, and a position-remapping routine. The other seventy-nine are float, several of them in matrix factorizations where the divide is a genuine pivot.

TWO PROOFS SHIPPED, matching the spirit of a modulo-sibling proof the check already had. A divide that IS the loop bound must be integer, because the for-condition compares an index against it. And a quotient later used as a slice INDEX must be integer — which the pre-existing direct-index guard could not see, because in both shipped instances the quotient passes through a variable first. Counted before shipping: eighty-nine to eighty-six, exactly the three integer sites, zero float sites lost, verified by spot-checking three known float sites that must remain.

THE OTHER SEVEN NEED TYPE INFORMATION and are not addressable in an AST-only analyzer: their quotients feed attribute structs or offset arithmetic rather than a bracket index. Every looser signal tried also swept up float sites — a first attempt treated the quotient being used again as the signal and wrongly flagged six, all of the shape quotient-then-multiply. Those seven are mitigated by the message, which now leads with the operand-type warning rather than a speedup range.

THE THRESHOLD QUESTION, cast as a rule. Three instances would not justify a performance narrowing, and I have declined two this campaign at similar counts. The asymmetry justifies acting here: suppressing a wrong recommendation costs nothing because the recommendation was wrong, and what is given up is a suggestion at sites where it would have produced zero. For a perf check the failure mode is low precision; for a correctness hazard low precision is what must be avoided and low recall is tolerable.

A VERIFICATION NOTE worth its own rule. One mutation appeared to leave a recall floor green, which reads as a toothless test; it had silently no-oped on a shell escaping problem and the floor was fine. Earlier in the campaign a different mutation landed on a dead duplicate of its target with the same misleading result. A green result after a mutation means nothing until the mutation is known to have applied.

## R-01KYRSZHMDEWA9J1T6PBQWXY56 Both floated narrowings for the register-blocking check were refuted by counting, a third form shipped at 96 percent precision, and disassembly vindicated the check's premise
kind: research
state: draft
created: 2026-07-30

Completes the triage's third-ranked item by doing the arithmetic it could not. The outcome is two rejections, one adoption, and a measurement that settles a doubt in the check's favour.

BOTH FLOATED PREDICATES FAIL, and the first fails in a way visible inside the hit set. Suppressing on the presence of any call expression removes twenty-eight of sixty-five and loses TEN of the forty-five genuine sites, because Go models a numeric conversion as a call expression - and seven of those casualties are the single-precision widening twin of a double-precision hit the same predicate keeps: same kernel, same file, differing only by a conversion around each load. An infinity constant likewise compiles to a constant load and no call, and would have silenced three more. The second predicate, suppressing when the re-read operand is a scalar, removes five and loses two genuine sites for two and a half points; it fires on one operand at five sites, two of which are textbook attention blocking the detector merely failed to NAME, because a range value can never witness the shared operand.

WHAT SHIPPED is the counted refinement: a call that is neither a conversion nor a builtin. Nineteen removed, eighteen of them false positives, thirteen being sites this repository itself labels a generic fallback for exotic dtypes that reaches every element through an accessor. Precision from 69.2 to 95.7 percent at 97.8 percent recall, with one named casualty where an infinity check sits in a mask bail-out and never in the inner product.

THE DISASSEMBLY IS THE MOST USEFUL PART, and it went the other way from expectation. The triage had doubted whether the reloaded operand survives register allocation, and the check's message now tells readers to confirm it. At the five most-genuine sites it survives every time - two or three scalar float loads per single fused multiply-add, base register hoisted but index reloaded each iteration. Five out of five for the check. At two of those sites the dominant per-iteration cost is not even the float load but a row slice-header reload plus bounds check, which the same unrolling would also amortize, so the stated payoff is if anything conservative.

TWO DEFECTS THE COUNTING SURFACED that no predicate addresses, and the hedged message does not cover: two hits recommend a transform that is ILLEGAL, because the flagged loop is itself the state-space recurrence, and two more recommend blocking loops that are ALREADY blocked by hand - one of them a hand-written kernel tail whose own comment says so. The existing stride exclusion cannot see either, because it inspects only the flagged loop's post statement rather than an enclosing one. Extending it is uncounted and not proposed.

A CORRECTION TO MY OWN TEXT. The message I wrote one iteration earlier claimed a body containing a call or a transcendental is bottlenecked elsewhere so the load is already free. False for conversions and for constant folds, which are the calls present at twelve of the twenty-eight call-bearing hits. Replaced.

AND A VACUOUS FLOOR CAUGHT BY MUTATION: the first version of the silence test used only accessor calls, which the detector cannot witness a shared operand through at all, so it passed for an unrelated reason and stayed green when the exclusion was removed. Rewritten to keep an index-witnessed operand alongside a real call.

## R-01KYRTB7R1F9EBYBGSKY7YE684 The register-blocking check now has zero false positives in its population, and the recount rule is what kept the last narrowing honest
kind: research
state: draft
created: 2026-07-30

Completes the precision work on the output-invariant-reload check. Two narrowings shipped across two iterations take it from 69.2 percent precision on 65 hits to 100 percent on 44.

THE RECOUNT MATTERED, WHICH IS THE POINT. A counting task had measured the final narrowing at three of the pre-filter sixty-five hits. Recounted against the forty-six that survive the earlier non-trivial-call exclusion it is TWO - one of the three had already been removed there. Acting on the stale figure would have overstated the change by half, and the rule requiring a recount against survivors rather than the original population is exactly what caught it. The two remaining turn out to be precisely the two false positives left, so the check goes from forty-four genuine of forty-six to forty-four of forty-four.

THE SIGNAL: a body declaring more than one scalar float accumulator is not the single-accumulator reload shape the check targets. Either the operand already feeds several accumulators, which IS the recommended transform - so recommending it again multiplies code paths, the hazard a sibling check reports - or the accumulators consume the operand differently and nothing invariant remains to amortize. The two sites are one of each: a hand-written vector-kernel tail already feeding four accumulators from one load, and a site whose two accumulators read the same operand through different indices. My first rationale called both already-blocked, which is true of the first and wrong about the second; the comment now distinguishes them.

WHY THE EXISTING STRIDE EXCLUSION CANNOT REACH EITHER: it inspects only the flagged loop's own post statement, while in both cases the blocking lives on an enclosing loop or in the accumulator set.

A SECOND VACUOUS FLOOR THIS WEEK. The silence test's first version strided by four, which the pre-existing stride exclusion already covers, so it passed without the new clause and stayed green when the clause was removed. Rewritten with stride one it goes red correctly. That is now twice in a few days that a floor passed for a reason unrelated to the clause it appeared to defend, and both were found only by confirming the mutation applied and then reading which tests moved.

STILL OPEN and recorded rather than acted on: two hits recommend a transform that is ILLEGAL, because the flagged loop is itself a sequential state-space recurrence. They are currently excluded for an unrelated reason - they call a transcendental - so the illegality is incidental to the fix rather than the reason for it. A dedicated sweep for checks that recommend illegal rather than merely unprofitable transforms was launched and died on a transient server error; it is worth re-running, because the two confirmed instances of that class so far were both found by hit-by-hit auditing and not by any predicate.

## ADR-01KYTPF84PEC0TP24AWYDG6HEY May reduction-accumulator splitting break a bit-exactness REGRESSION PIN (not an external contract) where it is profitable?
kind: adr
state: done
created: 2026-07-30
context: Splitting a serial float reduction into four partials is the largest optimization available in this tree: measured 3.03x at d=512 f64, 3.65x at d=768 f32, 2.90x on a fused dot-plus-norm pair, on M2 Pro darwin/arm64. It is currently unshippable everywhere. Every profile-reachable candidate is pinned by a bit-exactness test, and the integer subset that is exactly associative and needs no permission is 6 of 344 findings with no hot instance. Two pins are EXTERNAL CONTRACTS and must stand regardless: nlp turboquant (TurboQuant regenerates the matrix at dequantization, so a changed draw breaks already-quantized models) and backend/cpu mha_select (Float64bits parity against backend/ref, which would force a simultaneous change in a package carrying eleven open PRs). Three are REGRESSION PINS, written to prove an earlier refactor changed nothing rather than to guarantee a value to an external consumer: nlp contrastive_search (bit-for-bit vs the original serial reference, ~2.9x available on the second-hottest own-package line in nlp), autograd vjp_rwkv (TestWKVVJPBitIdentical golden checksums), classic ballTree.within (exact-label DBSCAN goldens; note this one is ALSO blocked independently because it reads its accumulator mid-loop to bail, so it is not a candidate either way). The question is only about the regression-pin category. Answering yes means the pinned checksums get updated and the goldens regenerated, and the guarantee weakens from bit-identical to numerically equivalent within rounding.
decision: yes for regression pins only: allow it where a measured win justifies regenerating the golden, contracts untouched
consequences: External contracts stay untouched: nlp turboquant, where TurboQuant regenerates the matrix at dequantization so a changed draw breaks already-quantized models, and backend/cpu mha_select, whose Float64bits parity against backend/ref would force a simultaneous change in a package carrying eleven open PRs. Regression pins may be regenerated where a benchmark justifies it. The guarantee weakens from bit-identical to numerically equivalent within rounding, so a pinning test must be CONVERTED to a tolerance comparison rather than deleted, or the property it protected is simply lost. First application is nlp contrastive_search MaxContextCosine.
status: accepted

kind: radio
option: no: keep bit-exactness everywhere; treat this class as closed and stop re-deriving it
option: yes for regression pins only: allow it where a measured win justifies regenerating the golden, contracts untouched
option: yes, but only for nlp contrastive_search as a single measured trial before deciding the general case
choice: yes for regression pins only: allow it where a measured win justifies regenerating the golden, contracts untouched

## T-01KYWWKJWRF06AEMTKY4A5FJEE T1019 max-normalized exp skip in WKV backward; perfscan PS3018
kind: task
state: draft
created: 2026-07-31
targets: autograd/vjp_rwkv.go, internal/perfscan/perfscan.go

MEASURED AND SHIPPED (commit 96367b21).

SITE. autograd/vjp_rwkv.go, the stabilized forward sweep of wkvChannelBackward. With z = math.Max(l, dgt), the code evaluated math.Exp(l-z) and math.Exp(dgt-z) and wrote EACH TWICE (once for the denominator dt, once for at and qd). math.Exp is not inlined, so the repeat was a genuine second evaluation and not a subexpression the compiler folds. Four calls per step where one suffices: the max guarantees one of the two exponents is exactly zero. The running-maximum fold carried a third instance — with positive decay b rises by w per step, so b[t] is the new maximum on nearly every step and exp(b[t]-curM) is exp(0).

RESULT. WKVScale128 198.0us -> 171.1us, -13.59%; WKVScale512 484.4us -> 412.3us, -14.87%; both p=0.000, n=12, interleaved.

BIT-IDENTICAL, and this is a real distinction rather than a tolerance claim: exp(0) is exactly 1, so substituting the literal changes no value, unlike an accumulator split which reassociates. Verified by dumping Float64bits of every gradient over five decay regimes (3.0, 0.7, 0.02, -0.5, 0) crossed with seq in {1,2,17,64,129} — 6540 values, byte-identical. Negative and zero decay are in the sweep because they are the regimes where b[t] is NOT the new maximum, so the fold guard takes its other branch.

NaN. The guard tests the max against the ARGUMENT (if z != l) rather than branching on the original comparison (if l >= dgt). With a NaN operand math.Max yields NaN, the equality then fails, and both exponentials still evaluate exactly as before. Branching on the comparison would substitute a 1 the original never produced.

GENERALIZED as perfscan PS3018 max-normalized-exp, which is what found this site. 22 remaining findings tree-wide, the rest in backend/cpu/wkv.go and backend/ref/wkv.go, both contested by open PRs (#692 perf-wkv-f32-cpu) and therefore left alone.

CHECK CONSTRUCTION, two things worth keeping. First, the initial version of PS3018 reported its OWN motivating fix as a defect: the applied form keeps math.Max and still contains the textual math.Exp(pp-q), so the detector matched it. The before/after validation is what caught this, not the floors — the floors all passed. A suppression for calls guarded by an inequality test against the max was added. Second, one clause was vacuously covered: the silent-without-max floor has no max at all, so a mutant accepting ANY max in scope stayed silent and the floor passed for the wrong reason. A distinct floor with a max PRESENT but a different subtrahend was added. All four clauses now redden exactly one floor under mutation: applied-form suppression, subtrahend-is-the-max lookup, argument match, subtraction requirement.

The check reports per CALL SITE rather than per exponent, deliberately — collapsing duplicates would have hidden half the cost here.

## T-01KYWWN4QGFKTRBJ4JKKGPCKRS T1019 max-normalized exp skip in WKV backward; perfscan PS3018
kind: task
state: done
created: 2026-07-31
targets: autograd/vjp_rwkv.go, internal/perfscan/perfscan.go

MEASURED AND SHIPPED (commit 96367b21).

SITE. autograd/vjp_rwkv.go, the stabilized forward sweep of wkvChannelBackward. With z = math.Max(l, dgt), the code evaluated math.Exp(l-z) and math.Exp(dgt-z) and wrote EACH TWICE (once for the denominator dt, once for at and qd). math.Exp is not inlined, so the repeat was a genuine second evaluation and not a subexpression the compiler folds. Four calls per step where one suffices: the max guarantees one of the two exponents is exactly zero. The running-maximum fold carried a third instance — with positive decay b rises by w per step, so b[t] is the new maximum on nearly every step and exp(b[t]-curM) is exp(0).

RESULT. WKVScale128 198.0us -> 171.1us, -13.59%; WKVScale512 484.4us -> 412.3us, -14.87%; both p=0.000, n=12, interleaved.

BIT-IDENTICAL, and this is a real distinction rather than a tolerance claim: exp(0) is exactly 1, so substituting the literal changes no value, unlike an accumulator split which reassociates. Verified by dumping Float64bits of every gradient over five decay regimes (3.0, 0.7, 0.02, -0.5, 0) crossed with seq in 1, 2, 17, 64, 129 — 6540 values, byte-identical. Negative and zero decay are in the sweep because they are the regimes where b[t] is NOT the new maximum, so the fold guard takes its other branch.

NaN. The guard tests the max against the ARGUMENT (if z != l) rather than branching on the original comparison (if l >= dgt). With a NaN operand math.Max yields NaN, the equality then fails, and both exponentials still evaluate exactly as before. Branching on the comparison would substitute a 1 the original never produced.

GENERALIZED as perfscan PS3018 max-normalized-exp, which is what found this site. 22 remaining findings tree-wide, the rest in backend/cpu/wkv.go and backend/ref/wkv.go, both contested by open PR 692 and therefore left alone.

CHECK CONSTRUCTION, two things worth keeping. First, the initial version of PS3018 reported its OWN motivating fix as a defect: the applied form keeps math.Max and still contains the textual math.Exp(pp-q), so the detector matched it. The before/after validation caught this, not the floors — the floors all passed. A suppression for calls guarded by an inequality test against the max was added. Second, one clause was vacuously covered: the silent-without-max floor has no max at all, so a mutant accepting ANY max in scope stayed silent and the floor passed for the wrong reason. A distinct floor with a max PRESENT but a different subtrahend was added. All four clauses now redden exactly one floor under mutation: applied-form suppression, subtrahend-is-the-max lookup, argument match, subtraction requirement.

The check reports per CALL SITE rather than per exponent, deliberately — collapsing duplicates would have hidden half the cost here.

## T-01KYWY6746EZETJRSB0QD5PTTV T1020 window the unrolled dot loop; perfscan PS3019; ballTree rejection
kind: task
state: draft
created: 2026-07-31

MEASURED AND SHIPPED (commit 53ea1cb6).

SHIPPED. nlp dotAndNorm. The loop is bounded by i+4 <= len(cand) and reads cand and ctx at four constant offsets each. That bound does NOT discharge the reads: i+4 can overflow, so the prove pass keeps a bounds check on every one of the eight. Cutting a 4-wide window once per iteration replaces them with two slice checks (compile-time count, 8 to 2), and the window length is a constant the compiler folds. MaxContextCosine_8x1024_d1024 621.2us to 520.9us, -16.15%; _64x512_d768 1.508ms to 1.228ms, -18.55%; geomean -17.36%, p<=0.001, n=12. Bit-identical — operands and their order are untouched, only the bounds proofs change.

REJECTED at the site that motivated the whole check. classic ballTree within, the L2 leaf test, has the identical shape and is 31.7% of the classic profile. Windowing took it from four checks per unrolled step to one. It measured -1.11% (p=0.004) — but DBSCANFitManhattan, which exercises the L1 arm that was not touched, moved -1.06% (p=0.010) in the same interleaved run. The treated cell and the untouched control are indistinguishable, so nothing was attributable and the change was reverted. Mechanism: that loop carries a data-dependent early exit whose misprediction already dominates (an earlier profile put that single branch at 450ms against 30ms for the arithmetic it guards), so the bounds checks hide behind it. The discriminator for this class is therefore a BRANCHLESS body with no loop-carried dependency; dotAndNorm has one, within does not.

An earlier run of the same A/B, taken while three research subagents were still executing, showed a geomean of +5.00% with per-cell spreads of 16 to 26% and no significance anywhere. It was discarded, not interpreted. Timing while agents run produces garbage in both directions.

GENERALIZED as PS3019 unrolled-index-not-windowed: 14 findings tree-wide. Predicate counted before the detector was written — 36 loops in six packages have the unrolled shape, of which 35 carry at least one live bounds check in the body; restricting to a len() bound (rather than a variable bound like hi or k) leaves the 14 provable ones. Four clauses are mutation-proven against exactly one floor each: lane count, unroll factor, the len-versus-cap test, and the loop-variable match.

TWO CONSTRUCTION FINDINGS. (1) An explicit this-base-was-already-sliced suppression was written and then DELETED. It could never fire on the applied form, because once the reads move onto the window they are indexed by constants rather than by an offset off the loop variable and nothing matches anyway; and on a HALF-converted loop, which still carries checks on the lanes not yet moved, it would have silenced a real finding. (2) Three of five floors initially passed for the wrong reason and were caught only by mutation — the windowed floor (no loop-variable offsets remained), the unroll-of-one floor (only one lane, so the lane clause suppressed it first), and the variable-bound floor (the bound was not a call at all, so a detector accepting any call stayed quiet). Each needed a fixture built AROUND the construct its clause discriminates, per A-SILENCE-FLOOR-MUST-CONTAIN-THE-CONSTRUCT-001, which this round exercised three more times.

ALSO TRIED AND FAILED: reordering the unrolled reads to touch the highest offset first, so that one check would establish the rest. All four checks remained. Only the explicit window works.

## R-01KYWY901EFR38Y8JTM5HTDDT2 R Unit surveys: classic/svm.go, vision, rl - profiled, ranked, mostly blocked
kind: research
state: draft
created: 2026-07-31

Three read-only unit surveys run in parallel, each profiling with -nodefraction=0 -nodecount=200000 (the default cutoff has produced false floor conclusions here). Findings are CANDIDATES; nothing below is measured by the main agent except where stated.

VISION - NEGATIVE RESULT, the useful kind. Summed flat samples across every vision source line total 0.056% of the profile. Every function in the package is a dispatcher that builds index slices and calls backend.Execute; the CPU lands in backend/cpu and the runtime. The ceiling on any vision-local CPU optimization is therefore 0.056%, and none should be attempted. 152 bounds checks remain in the package and are worth nothing for the same reason. Two real items: (a) vision.go, mae.go and vlm.go - 916 of 3329 non-test lines, 27.5% - have NO benchmark at all, CNN.Forward being the notable gap since convolution is the one genuinely expensive per-image primitive; nothing can be claimed about them on this host until that exists. (b) swinPatchify duplicates the body that patchifyImage consolidated for ViT, MLPMixer and MAE - Swin was left behind. Routing it through patchifyImage is bit-identical (pure data movement, and the f32-f64-f32 round trip it removes is an exact identity) and is justified as duplicate-code removal, NOT as a perf win, since the expected delta sits below benchmark noise.

RL - one strong finding, verified INDEPENDENTLY by the main agent at compile level. rl/continuous.go:327, the Polyak softUpdate inner loop, carries two bounds checks (confirmed at 327:20 and 327:36, plus the same pair at 337:36 and 337:61 in the F32 arm) AND recomputes the loop-invariant 1-tau every iteration - FMOVD of the constant 1.0 followed by FSUBD appears inside the loop body in the -S dump, also confirmed directly. The mechanism claim is the interesting part: LICM fails to hoist 1-tau BECAUSE the two bounds-check branches split the body into separate basic blocks, and Go SSA will not hoist across a block that can panic. The bounds checks therefore cost more than their own instructions - they block an unrelated optimization. Six of fourteen instructions in the body are overhead, and the loop is 72% of BenchmarkSoftUpdate flat time. NOT YET MEASURED. Caveat that must be respected: rl/softupdate_parity_test.go documents a real 1-ulp failure caused by FMA contraction differing between expression shapes, so merging basic blocks can move which multiply fuses; ship only if the tolerance-zero parity test passes AND the post-change dump still shows one FMADDD in the same position. Everything else in rl is blocked by bit-exact determinism contracts (step_paths_parity_test.go digests actions, rewards, dones and logpOld with an xor of Float64bits specifically to catch reordered summation; ppo_test.go pins whole learning curves; gae.json is a committed golden), which kills every reassociation candidate outright. tabular.go has no benchmark.

SVM - the hottest line is NOT the best target, and that reframe is the finding. svm.go:178, the RBF squared-distance inner loop, is 22.5% of a single-core profile, but it sits inside kernelCache.column which is ALREADY parallelized; reconciled to 12 cores it is roughly 13% of Fit. The SMO selection scans are 100% serial and become 30-35% of Fit at 12 cores. Ranked candidates: (1) memoize the SMO membership predicates at svm.go:395-435 - the disassembly shows the two-clause disjunction expanding to twelve FCMPD, eight of them the identical y[t] > 0 test, on a coin-flip predicate, so the cost is branch misprediction; alpha changes at exactly two indices per step, so boolean masks are O(1) to maintain and the transform is pure predicate memoization with no arithmetic change. (2) svm.go:484, three bounds checks in the gradient update, four of fifteen instructions, with no loop-carried dependency to hide them - the highest-confidence item. (3) DecisionFunction uses a static equal split via parallelBuild rather than the claiming parallel.Rows the same file already uses for columns; the alloc win is certain (14-21 allocs per op scaling with GOMAXPROCS against 2 for parallel.Rows) but the agent explicitly RETRACTED its own efficiency-core imbalance hypothesis after an interleaved A/B showed within-configuration spread dwarfing the difference. (4) the RBF kernel diagonal is provably exactly 1.0 and need not be computed. BLOCKED: every reassociation, by svm_parallel_parity_test.go, which pins hex digests of the fitted model on the exact data the benchmark uses.

CROSS-CUTTING: all three profiles are dominated by scheduler and GC frames (73% in rl, 74% in svm at 12 cores, 68% in vision) originating in the backend/cpu pool wake-and-spin path and the tensor allocator. That is the largest single lever visible from any of these units and it lies outside all of them.

## T-01KYWZ0378FEJBKYXSHV0J75HR T1021 bounds check blocking LICM in rl softUpdate; perfscan PS3020
kind: task
state: draft
created: 2026-07-31
targets: rl/continuous.go, internal/perfscan/perfscan.go

MEASURED AND SHIPPED (commits 7d16b85d and 5818e71b).

SITE. rl/continuous.go, the Polyak soft update inner loop, both the F64 and F32 typed fast paths. Each iteration read so[j] and to[j], and each read carried a bounds check.

THE MECHANISM, which is the durable part. The two bounds checks are panic edges, and a panic edge splits the loop body into separate basic blocks. Go SSA will not hoist across a block that can panic, so the loop-invariant (1-tau) was rematerialized on every iteration - FMOVD of the constant 1.0 followed by FSUBD, both visible in a -gcflags=-S dump. The bounds checks therefore cost strictly more than their own instructions: they also trapped an unrelated optimization. Ranging over the destination chunk bounds the index and cutting the source to the same length relates the second operand, which discharges both checks into two per-chunk slice checks hoisted above the loop, and only then does the invariant hoist. Body went from 14 instructions to 5.

RESULT. SoftUpdate 50.83us to 40.00us, -21.30%; SoftUpdateF32 51.47us to 41.88us, -18.62%; both p=0.000, n=12. RLForward, which the change cannot touch, was flat at p=0.165 - the sibling-arm control from AN-UNTOUCHED-SIBLING-ARM-IS-THE-CONTROL-001, applied on its first opportunity. Allocations unchanged.

BIT-IDENTITY NEEDED CHECKING RATHER THAN ASSERTING. rl/softupdate_parity_test.go records a real 1-ulp failure caused by FMA contraction differing between two expression shapes, and merging basic blocks can move which multiply the backend contracts. The post-change disassembly still shows exactly one FMADDD in the same position fusing the same multiply, and TestSoftUpdateParallelBitIdentical at tolerance zero passes.

NEW BENCHMARK. BenchmarkSoftUpdateF32. The F32 arm carried the identical two defects and had no benchmark at all, so the symmetric fix could not have been validated; it is now measured rather than argued from symmetry.

GENERALIZED as PS3020 invariant-behind-bounds-check. PREDICATE DISCIPLINE decided the shape: a first predicate accepting any loop-invariant arithmetic in an indexing loop found 64 sites, nearly all integer addressing terms that fold into an addressing mode and cost nothing. Requiring the invariant to be an operand AGAINST AN INDEXED READ - a value the loop recomputes rather than an address - cut it to 2, both the same coefficient shape as the motivating site: backend/cpu crossentropy (1-eps) and nn soap (1-b2). Neither is in a free lane and neither was touched.

The check catches both pre-fix arms and is silent on both shipped ones. Two clauses are mutation-proven: dropping the pairing test reddens the addressing floor, dropping the invariance test reddens three floors. The range-loop floor documents a STRUCTURAL exclusion rather than a predicate and now says so explicitly - a range loop is an ast.RangeStmt, filtered by the type assertion before any clause runs, and blanking the counter test leaves that floor green.

A TOOLING BUG WORTH REMEMBERING. The first predicate counter reported zero sites tree-wide. It was not that the class was absent: the ast.Inspect visitor treated the end-of-children nil node as an unknown leaf and set its invariance flag false for every expression, so nothing could ever match. It was caught only by running the counter against the KNOWN pre-fix source and demanding it fire. A detector that reports nothing must be proven able to fire before its zero is believed.

## R-01KYWZYENEFB7R5R1SANDAJBFB R Unit surveys round 2: linalg, autograd, classic clustering - mechanical evidence only
kind: research
state: draft
created: 2026-07-31

Three read-only unit surveys, all briefed to report mechanical evidence only (profile SHARE, disassembly, bounds-check and allocation counts) and explicitly forbidden from reporting A/B timing deltas, since they ran concurrently and wall-clock comparisons between them are contaminated. All three complied. Validation stayed in the main agent.

LINALG - the highest-value finding of the session, since acted on and shipped as T1022. The reframe: Lstsq received a free-dimension column jam and LU.Solve plus CholSolve have byte-for-byte the same loop shape, the same solveCols scaffolding, the same bit-identity argument and the same test coverage, and did not get it. Together those two are 51.7% of the package profile. Every mechanical claim was independently re-verified in the main agent before acting: colJam=4 exists in qr.go for Lstsq only; TestSolveBitStableGoldens pins with a Float64bits digest; LU's substitution bodies carry ZERO bounds checks (the ones the scanner reports at those lines are hoisted preamble slice ops); cholesky.go:143 carries exactly one and svd.go:48-49 exactly two. STILL OPEN from this survey: the same jam for CholSolve (16.73% share, and it additionally amortizes one surviving bounds check and a two-load slice-header walk across four FMAs); a flat companion for cholFactor, whose slab already exists and is currently discarded; and the Factor parallel branch allocating a closure per elimination step, which shows as roughly 1034 extra allocs at n=768 and is a resource-axis item only.

AUTOGRAD - three candidates, none yet measured. Strongest is vjp_logdet: solve the forward substitution directly in column-major and delete both the linv workspace and the O(n-squared) transpose pass that follows it, which also removes false sharing in the parallel arm, since today worker j writes a COLUMN whose stores share cache lines with every other worker. Second is vjp_conv1d, where an invariant t-(K-1) is rematerialized behind a sign-test guard that is false for only 3 of 2048 positions - the loop HEADER profiles higher than either FMA line. Third is exploiting the symmetry already proved in-tree at vjp_solvespd for the Cholesky S block, a 2x flop reduction. IMPORTANT NEGATIVE: 97.8% of all allocated objects in the package profile come from a TEST-ONLY pre-existing baseline function, which would dominate any naive reading of the memory profile. Also: 52 of 77 registered VJPs have no benchmark at all, vjp_ssm.go being the largest gap by code mass, so nothing can be claimed about them here.

CLASSIC CLUSTERING - three candidates, none yet measured. GaussianNB.Fit is the cleanest instance yet of the bounds-check-blocks-hoisting mechanism: 14 of 19 instructions in the innermost loop are invariant address arithmetic trapped by four panic edges. GMM full-covariance solve carries eight live bounds checks in an O(d-squared) kernel, and the disassembly shows the damage is register pressure rather than the checks themselves - eight invariant length values stay live, which spills the loop induction variable and reloads a slice pointer every iteration. DBSCAN neighbor lists are the top production allocation site at 23884 objects. Also confirmed independently: the ball-tree windowing change rejected last round is still mechanically wrong to retry, and the agent added a reason worth keeping - a PARTIAL bounds-check fix cannot win there, because one surviving panic edge keeps the function non-leaf and preserves its frame and stack-growth check. KMeans has real allocation churn and zero benchmarks, so it was correctly not proposed.

CROSS-CUTTING, unchanged from the previous round: every profile is dominated by scheduler and GC frames originating in the backend/cpu pool and the tensor allocator. Both the linalg and classic agents flagged solveCols and the DBSCAN static chunk split as structurally suspect but explicitly DECLINED to rank them, on the grounds that parallel-efficiency numbers taken while sibling agents saturate the cores are exactly the contaminated measurement they were warned about. That restraint is the correct call and is why those two remain open.

## T-01KYX0YF6XEGVAZC28FAP7YX1H T1023/T1024 GaussianNB row hoist, conv1d guard bounds; PS3021 + PS3016 amendment
kind: task
state: draft
created: 2026-07-31
targets: classic/naivebayes.go, autograd/vjp_conv1d.go, internal/perfscan/perfscan.go

MEASURED AND SHIPPED (commits 0e2aad7b, 8f1fd095, 697a830b).

T1023 GAUSSIANNB.FIT. Both O(n*d) accumulation loops indexed theta[c][j] and x[i][j] directly - three bounds checks in the mean loop, four in the variance loop. Cutting the class row and the sample row to one length lets range prove every index. GaussianNBFit/n4000 430.2us to 292.3us, -32.07%; /n16000 1.449ms to 1.040ms, -28.24%; both p=0.000, n=12, with GaussianNBPredict and PCAFit flat as controls. Bit-identical; the committed scikit-learn golden passes unchanged.

WHY IT PAID 15x WHAT THE SAME TRANSFORM PAID ON CHOLESKY, which is the transferable part: the checks were not merely costing their own instructions, they were BLOCKING A HOIST. Each is a panic edge splitting the body into its own basic block, and Go SSA will not hoist across a block that can panic, so the class offset times the row stride, the sample offset times the row stride, and BOTH slice headers were rematerialized on every inner step although all four are invariant in j - 14 of the 19 instructions were address arithmetic that should have been loop-invariant. Rank a two-deep site by how much invariant work is trapped behind its checks, not by the check count. This landed as an amendment to PS3016 rather than a new rule, because PS3016 already fires on the pre-fix site; it was verified to fire before anything was designed.

T1024 CONV1D BOUNDARY GUARD. The innermost tap loop ran a per-tap `j >= 0` where j is linear in the tap index. The guard is false for only the first K-1 of L positions - 3 of 2048 at the benchmark shape - and being a branch it also stopped t-(K-1) from hoisting although it is invariant in both inner loops. The profile showed the consequence directly: the loop HEADER was a larger share than either line the guard protected. Computing the first valid tap once per position and starting the loop there removes both costs.

REPORTED AS MEASURED, NOT AS HOPED. Conv1DBackwardF32 8.659ms to 7.973ms, -7.92% (p=0.000, n=16). The F64 arm was directionally -4.7% and did NOT reach significance at n=16 (p=0.210). An earlier n=12 run had shown the F64 arm at p=0.020; that did not survive once the new F32 cell shared the run and the spread widened, which is why it was re-run rather than banked. Shipped for both arms regardless, because the transform strictly removes instructions from the inner loop - a branch per tap plus the rematerialized invariant - so there is no mechanism by which it could regress, and leaving one arm on the slower form to match a null result would be worse. BenchmarkConv1DBackwardF32 is new; that arm had no benchmark at all.

GENERALIZED as PS3021 monotone-guard-in-loop. Three findings tree-wide plus the two it was built from. The density is the deliberate result: the neighbouring free-dimension predicate attempted last round matched 117 sites and was withheld as noise, and restricting this one to ADDITIVE movement with a FIXED comparison side is what makes it usable. Five clauses mutation-proven against distinct floors. The invariant-side clause first passed vacuously - the both-sides-move floor suppresses earlier - and needed a fixture whose bound is a PRODUCT of the loop variable, which the additive test rejects so the comparison looks one-sided while the bound plainly moves. That is the fourth consecutive round in which a silence floor passed for the wrong reason until a fixture was built around the construct its clause discriminates.

The advice carries the null result beside the win, and warns about DIRECTION, which is how this rewrite becomes a silent bug: a guard true for a prefix bounds the loop above, one true for a suffix bounds it below, and reversing them drops real work with no test necessarily noticing.

LANE COLLISION TO RESOLVE, recorded because it is a live risk and an error of process. T1022 changed linalg LU.Solve and svd.go without a lane check, and open PRs 651 (flat row-major LU factor) and 663 (SVD V-accumulator) touch both. 651 is not a mechanical conflict: it converts the factor from a slice of rows to a flat buffer, which the column jam indexes. PR 652 rewrites CholSolve, so the planned CholSolve jam - the direct sibling of T1022 and the highest-value item left in linalg - is blocked until that lands. Lane checks were run for rl and svm that round and skipped for linalg.

## T-01KYX1R8MCEN5TTV3B9XZG73N8 T1025 logdet solves in the consumed layout; perfscan PS3023
kind: task
state: draft
created: 2026-07-31
targets: autograd/vjp_logdet.go, internal/perfscan/perfscan.go

MEASURED AND SHIPPED (commits dafcecf3, 46a1f512).

SITE. autograd vjp_logdet. The forward substitution built the triangular inverse row-major, then a separate O(n^2) pass transposed it because the A-inverse contraction needs columns. Solving directly into column-major deletes the transpose pass, the whole intermediate and n allocations, and removes two further costs a line profile does not show: the substitution inner loop stopped walking down a column of a slice-of-slices, losing a row-pointer load and a bounds check per element (the accumulation loop now carries no bounds check at all); and FALSE SHARING went away, because solving row-major had worker j writing a COLUMN, so worker j+1 stored into the adjacent eight bytes of the same rows and every write contended for a cache line. Each worker now owns a contiguous row.

RESULT. LogDetBackward_512 31.10ms to 27.87ms, -10.37%; _256 3.910ms to 3.639ms, -6.93%; both p=0.000, n=12; allocs/op -31.93% and -30.22%. CholeskyVJP_64 flat as the untouched control (p=0.887, allocations byte-identical).

BIT-IDENTICAL, VERIFIED RATHER THAN ASSUMED. This VJP is pinned only at tolerance (gradcheck rel 1e-4, closed form 1e-8), so a Float64bits dump of every gradient across n in 3, 8, 17, 33, 64 was compared before and after: 5547 values, byte-identical. cv[t] is col[j+t] = Linv[j+t, j], exactly the lv[t][j] the previous form read, in the same ascending order.

GENERALIZED as PS3023 transpose-pass-over-built-matrix, and the interesting part is WHY it is a separate check rather than an amendment. PS1010 already covers column walks over slices of slices, and it deliberately DECLINES this shape: it reports only when the inner loop assigns to something free of the inner variable, because then interchange is the remedy, and its own comment records that a transpose writes the inner variable on the left and strides whichever way it is run. The excluded case has a remedy of a different kind — not reorder the copy but delete it, by having the producer emit the consumed layout. PS3023 targets exactly that complement. One finding tree-wide besides the motivating site (linalg svd.go:140, contested by an open PR, untouched). Three clauses mutation-proven.

TWO PROCESS FAILURES, recorded because both were caught by existing rules and both would otherwise have shipped bad work. (1) The first draft of the predicate targeted column walks in general and matched 40 sites — squarely PS1010's territory. It was designed from scratch WITHOUT first running the full scan against the pre-fix site, and the duplication surfaced only when the new helper's name collided with the existing one. The round before, the same step was run for the guard check, found nothing, and correctly justified a new rule; skipping it here nearly shipped a duplicate. The lesson is not new — PERF-CHECK-EXISTS-BEFORE-ADDING-ONE-001 already says it — it is that the check must be run BEFORE the predicate is designed, not after. (2) The narrowed detector then reported zero findings tree-wide INCLUDING on its own motivating site, because it tested the transpose write with mentionsAsValue, a helper written for a different check specifically to EXCLUDE index positions, and a transpose destination is indexed by the inner variable. Only the rule requiring a zero-finding detector to be proven able to fire caught that.

## T-01KYX21XAWESHAQYZ8A8WW6P8D T1026 GMM full-covariance solve operands cut to one length; PS3017 amended
kind: task
state: draft
created: 2026-07-31
targets: classic/gmm.go, internal/perfscan/perfscan.go

MEASURED AND SHIPPED (commits 786200fe, 7ca5b94b).

SITE. classic GMM logGaussianFullBatch, the O(d^2) triangular solve. Four L rows and four y vectors were indexed directly inside a loop written as a range over an INTEGER, which bounds nothing and proves no relation to any slice, so all eight reads kept a bounds check. The damage was not the checks: it was register pressure. Eight loop-invariant lengths stayed live simultaneously, which spilled the loop induction variable itself and forced a slice pointer to be reloaded every iteration. Cutting all eight to one length and ranging over the first discharges every check in the inner accumulation - 8 to 0 in the 4-jam arm, 4 to 0 in the 2-jam arm.

RESULT. GMMFitFull 17.36ms to 16.45ms, -5.27% (p=0.000, n=12). GMMPredictProbaFull_2048x8_d32 -1.45% (p=0.033). The diagonal arm, which this change cannot touch, was flat (p=0.630).

CAVEAT THAT QUALIFIES ONE OF THOSE NUMBERS. The PCAFit control, also untouchable by this change, moved +0.81% (p=0.017) in the same interleaved run. That is systematic drift of nearly a point, so only the GMMFitFull result sits clearly outside it; the d32 cell should not be quoted on its own. Recorded because a reader comparing future GMM numbers needs to know one of these two is soft.

BIT-IDENTICAL, and gated for real rather than by tolerance: TestGMMFullBitIdenticalToGolden passes unchanged.

NO NEW SCAN RULE, deliberately, and this is the second withholding this session. Following SCAN-THE-PRE-FIX-SITE-BEFORE-DESIGNING-A-PREDICATE-001, the full scan was run against the pre-fix site FIRST: only PS3010 fires there, and its advice is the reassociation one, which the bit-exact golden forbids outright. So the actual defect is uncovered - PS3017 is the closest rule and it deliberately requires ranging over a SLICE, because a loop over an integer is syntactically identical without types and proves nothing (RANGE-OVER-AN-INT-LOOKS-LIKE-RANGE-OVER-A-SLICE-001). A dedicated check for integer-bounded loops with many indexed companions was built and measured at a four-operand threshold: 141 sites, nearly all of them the ordinary outer loop that indexes several parallel arrays, which is not a defect - and the predicate did not even isolate the inner loop that was. Withheld, with the number recorded so it is not retried blindly.

The learning landed as an amendment to PS3017 instead: rank a site by its OPERAND count rather than its check count, since each surviving check pins a live length; and the advice now names its own blind spot so a reader knows integer-bounded loops must still be found by reading.

## R-01KYX31FP2FH2TGBH5BDR0DN7T R tensor survey: compute at its floor, allocation is the axis, top candidate was a known rejection
kind: research
state: draft
created: 2026-07-31

A read-only survey of tensor/, a package that had never been surveyed and sits under everything else in the repo.

VERDICT ON THE COMPUTE SIDE: at its floor. No tensor line appears in a caller-heavy profile at all; the highest line in the package's OWN suite is 1.35%, and that share is an artifact of which microbenchmarks exist rather than of any real workload. The gather inner loop disassembles to six instructions with the bounds check folded onto the loop back edge. All six perfscan candidates were examined and none survives: two flagged reductions accumulate INTEGERS over two to four iterations (integer addition is associative, so the bit-identity warning does not even apply, and there is no latency chain at that trip count), one is a false positive on a loop that also computes strides, and two are cold panic-formatting paths.

THE REAL AXIS IS ALLOCATION COUNT PER TENSOR CREATION. tensor/ accounts for 86.5% of every object a Jamba decode step allocates. New costs four allocations: the tensorBlock, the Storage, the interface box Storage.data pays to hold its slice, and the data buffer itself. Byte arithmetic confirms the count exactly - 112 + 48 + 24 + 256 = 440, matching the reported 440 B/op.

THE TOP CANDIDATE WAS A TRAP, AND CATCHING IT IS THE MAIN RESULT. The survey proposed folding the Storage into tensorBlock as its highest-ranked item, correctly observing that the struct comment already claimed this was done while the code had no such field. That change had already been tried and MEASURED WORSE in an earlier working session: allocations fell about 23% and the decode step ran 6.86% slower, because Reshape and friends build a new Tensor over the SAME Storage, so one surviving view would pin the entire block instead of a bare 48-byte Storage. The measurement lived only in a session transcript, so nothing in the repository could stop a reader repeating it - and a reader promptly did. Fixed by correcting the comment to state what ships and why the Storage is excluded (commit 4210c32e), and by recording FOLD-ONLY-OBJECTS-WITH-IDENTICAL-LIFETIMES-001. An allocation count is a proxy, and it fails exactly when the removed object outlives the one it was folded into.

STILL OPEN AND UNMEASURED. (1) The view constructors - Slice, Transpose, Permute - each pay two allocations, a copyShapeStrides array plus the Tensor, and could take the same block treatment New already has. This is NOT the rejected change: a view block holds the view's own Tensor and its own shape/stride buffer, whose lifetimes are identical by construction, so the retention argument does not apply. Three benchmarks already report allocations. Note the Permute cell implies rank 4 while maxInlineRank is 2, so the inline size is a byte-for-coverage trade to sweep rather than guess. (2) Storage.data is an interface, so every allocation pays a runtime slice-to-interface box - 14% of objects in a decode step. An additive unexported typed-allocator interface would avoid it without touching the exported Allocator signature, but it grows Storage from 48 to about 104 bytes, trading bytes for objects; that one needs an A/B on both heap and pooled devices rather than an argument. A variant using unsafe.Pointer avoids the byte growth and was explicitly flagged as higher risk in the repo's foundational type.

COVERAGE GAP WORTH NAMING: New on a POOL-backed device has no benchmark. Only Pool.Alloc and Free in isolation are measured. The pool path is where the interface box is proportionally most expensive, since the buffer is recycled and the box never is.

## T-01KYX4B549E8Y9K4H63420H7B5 T1030 MHA routed through pooled exec helpers; perfscan PS3024
kind: task
state: draft
created: 2026-07-31
targets: nlp/mha.go, nlp/decode.go, internal/perfscan/perfscan.go

MEASURED AND SHIPPED (commits aab765b1, 4344f303).

SITE. nlp (*MHA).exec was a byte-for-byte clone of gpt.go's exec1 that never touched its receiver. Its thirteen call sites therefore allocated a fresh variadic slice on every backend dispatch, while exec1a, exec2, exec3 and exec4 - recorder-guarded, sync.Pool-backed, written for exactly this inference hot path - sat unused beside it. Five of the thirteen are on the per-token decode path. Every site now takes the pooled helper matching its arity; the one genuinely variadic site, an OpConcat over a parts slice, takes exec1 directly. The clone is deleted so nothing can drift back.

RESULT. GPTGenerate500RowBuf 235.3k to 225.3k allocs/op, -4.26% (p=0.000, n=10) - ten thousand fewer allocations per generate. KVCacheGrowthRowBuf allocs unchanged (p=0.595). NucleusTopP, untouched, identical allocations and flat time.

TIME IS NOT CLAIMED. sec/op came out -1.7% at p=0.143, which does not separate. An earlier n=12 run had shown -2.47% at p=0.007 and it did not reproduce at n=10, so it was not banked. These benchmarks are 69% backend worker-pool park and wake, which is why allocs/op is the metric this change is answerable for.

Bit-identical: the pooled helpers pass the same tensors in the same order to the same backend.Execute, and each defers to the variadic form when a recorder is attached, so the taping path is untouched.

GENERALIZED as PS3024 fixed-arity-variadic-call. Clean before and after on its own site: 12 findings pre-fix, 0 after. Twelve and not thirteen is correct - the remaining site is a real spread, which cannot avoid building a pack. Three clauses mutation-proven, and the pooled-helper suppression is the load-bearing one: a pooled helper hands its fixed arguments to the variadic form exactly when a recorder is attached, so without it the check reports the fix as the defect.

450 FINDINGS TREE-WIDE, ALL IN nn/, and the number is stated rather than tuned away. This is NOT the noise that got two candidate rules withheld earlier this session at 117 and 141 sites; those were mixtures of non-defects. This is one defect class in one package - nn ships nnIns1Pool through nnIns3Pool and 43 of its own wrapper clones bypass them - so the remedy already exists there and the count is a backlog rather than a false-positive rate. nn is not this agent's lane, which is precisely why the finding belongs in a scanner instead of a patch.

HOW THE CLASS WAS FOUND, worth keeping as method. A whole-tree duplicate-body scan (function bodies rendered with parameters renamed positionally, so a method and a free function still match) reported 44 clone groups, the largest being 29 identical exec methods in nn. Raw duplication was rejected as a rule - most Go clone groups are legitimate per-type boilerplate in the absence of generics - but it was the right INSTRUMENT for locating the class, and the shippable predicate turned out to be at the CALL SITE rather than the declaration.

TWO SELF-INFLICTED FAULTS, both caught by existing rules. The detector first reported zero findings everywhere, including its own motivating site: the check table entry was never added, so the category mapped to no ID and every finding was dropped. Then with exec1 on the configured list it reported 770, because the pooled helpers' own correct fallback calls matched. Narrowing the list and suppressing calls made from inside a pooled helper took it to the real 450.

## T-01KYX4YA4MF70VKH69NS1MX82K T1032 DBSCAN neighbour lists slab-allocated; PS2008 advice extended
kind: task
state: draft
created: 2026-07-31
targets: classic/dbscan.go, internal/perfscan/perfscan.go

MEASURED AND SHIPPED (commits 11ebdb60, 03861c42).

SITE. classic DBSCAN gave every core point its own make for its neighbour list - the single largest production allocation site in the package. They are now carved from a per-goroutine block bump allocator: one allocation per 4096-int block instead of one per core point, each list cut three-index so its capacity ends at its own last element, and a list longer than a block getting its own exact-sized one so the slab never forces a copy.

RESULT. DBSCANFit/eps4 allocs 3.515k to 1.189k, -66.18%, time -5.90%. DBSCANFitManhattan/eps16 allocs 4.836k to 1.251k, -74.13%, time -7.48%. Both p=0.000, n=12. Better than predicted - this was expected to move allocations only.

TWO CONTROLS, ONE OF THEM FREE. KNNPredict was flat on both axes. More usefully, the eps2 arm of the DBSCAN fixture is degenerate all-noise, so NO point is core and no list is ever built: it stayed flat too, which is direct evidence the win comes from the lists and nothing else in Fit. A fixture that already contains a degenerate arm hands you a control for free.

THE LIFETIME TEST WAS CHECKED, NOT ASSUMED. A block pins every row in it, so slabbing is only safe when the rows die together. Here neighbors is a local in Fit, never returned and never stored on the receiver, read only by the flood fill - so every core list drops at the same instant. Same test that made the tensor view block a 26% win and the tensor Storage fold a 6.86% loss.

NO NEW SCAN RULE - fifth withholding of the session, second for emptiness rather than density. The pre-fix site was scanned first: PS2008 is the slab check and it correctly stays SILENT, because it requires a loop-invariant length and a uniform slab needs uniform rows. So the varying-length case is genuinely uncovered. A predicate for the clone idiom was built and validated to FIRE on the pre-fix site, then found ZERO other instances tree-wide. One occurrence, now fixed, does not earn a detector.

The learning went into PS2008's advice instead, which is where a reader at such a site already is: the invariant-length requirement is real but not the end of the story, the complement remedy is a block bump allocator, the shape that hides these sites is `dst[i] = append(make([]T, 0, len(src)), src...)` with the make nested inside an append, and the lifetime precondition applies to both forms.

## ADR-01KYX550H2F2JAT4JHRCCBX2MG Remove the Storage.data interface box - typed fields, unsafe.Pointer, or leave it?
kind: adr
state: submitted
created: 2026-07-31
context: Storage.data is an any, so every tensor allocation pays a runtime slice-to-interface box: 24 bytes and one object that NEVER recycles, even on a pooled device. Newly measured with BenchmarkNewOnPooled (commit b0d31489): on the pool path a tensor costs 184 B in 3 allocations - block 112, Storage 48, box 24 - and a 256 KB tensor costs only 13 bytes more, because the buffer recycles and the rest is fixed overhead. The box is about 14% of all objects a decode step allocates. Removing it means changing the foundational type, and the two ways trade differently, which is why this is a decision and not a task. The exported Allocator.Alloc(dtype, n) any signature stays unchanged either way, and no package in this repo implements Allocator externally - all backends return tensor.Heap() - so an additive unexported fast path is safe.
status: proposed

kind: radio
option: A typed fields: an unexported typedAllocator (allocF32/allocF64/allocU16) on heapAllocator and Pool, typed slices in Storage. No unsafe. Storage grows 48 to about 104 bytes, so the pooled small case goes 184 B in 3 allocs to about 216 B in 2 - objects down a third, bytes up about 17%.
option: B unsafe.Pointer: data becomes an unsafe.Pointer plus the existing dtype and n; F32/F64/U16 become unsafe.Slice. Storage SHRINKS 48 to 40 bytes, so objects AND bytes both improve with no trade. Cost is unsafe in the most foundational type, on the path every tensor takes. Precedent exists in internal/npy, backend/cpu and backend/metal, but this is a different risk class.
option: C leave it: the box is real but bounded, and Storage is read by every accessor in the package, so neither rewrite is free to review.

## T-01KYYVBKHMESV9SJ4YPG01TSQQ T1034 merge campaign: 49 of 52 PRs merged, gofmt regression fixed, CUDA chain serialized
kind: task
state: draft
created: 2026-08-01

MERGE CAMPAIGN, run on explicit instruction to merge every open PR and then land this session's work as one PR on top.

RESULT SO FAR: 49 of 52 merged. 3 remain (681, 682, 683), all CUDA, all blocked only on serialization rather than on anything unresolved.

WHAT WENT SMOOTHLY. 47 merged in three batches, oldest first, each needing `gh pr ready` because 51 of the 52 were drafts. Merging in ascending order minimized conflicts but did not eliminate them: 5 PRs conflicted as main advanced under them.

THE gofmt REGRESSION, worth recording because it will recur. PR 684 existed precisely to fix a whole-tree gofmt failure on main, and the batch merged right after it REINTRODUCED the same failure in 6 files (from 648, 667, 668, 669, 675). The repo preflight runs gofmt over the whole tree, so main was red and every subsequent push was blocked by the pre-push hook. Fixed by PR 702, the same way 684 did it. The lesson is that a formatting gate which only runs in the pre-push hook and CI, and not in the PR merge gate, will be reintroduced by every batch that does not pass through a local push.

CONFLICT RESOLUTION, two shapes. (1) PR 649, cpu vexp: main had gained softcapF32/vsoftcapF32 from 648 and the PR added softplusF32/softplusF32x8 at the same place. Additive in vexp.go, but in vexp_amd64.go the conflict region cut THROUGH a function body - both sides ended inside a for loop and shared the closing braces after the marker - so naive concatenation produced a file that still compiled on the host and was only caught by gofmt, because vexp_amd64.go is not built on darwin/arm64. Verified afterwards with an explicit GOARCH=amd64 cross-build, which is the only way a formatting or syntax slip in an arch-gated file surfaces on this host. (2) All four CUDA PRs conflicted on ONE line: a long static CUfunction declaration list, each PR adding a different symbol. A scripted union resolver merges them by inserting each new symbol after the same predecessor it follows on its own side, then asserts no symbol was lost. Refuses to act unless both sides are the single-line declaration shape.

WHY THE LAST THREE ARE SERIAL. They all edit that same declaration line, so each merge re-conflicts the others. The chain is rebase, push, wait for CI, merge, repeat. Auto-merge is disabled on this repository, so none of it can be queued.

CUDA CANNOT BE VERIFIED ON THIS HOST - darwin/arm64, no NVIDIA toolchain - so for those four the local gates are gofmt, the non-cgo build and vet, and CI is the real verifier. 679 was merged only after its full check rollup came back green.

ENVIRONMENT DEFECT HIT AND REPAIRED. Mid-campaign every git command in the working worktree began failing with a work-tree error, including in an unrelated sibling worktree. Cause: the main repository is bare and worktrees rely on extensions.worktreeConfig, but this worktree's config.worktree had lost its core.bare=false, so it inherited bare from the main config. Repaired with a scoped `git config --worktree core.bare false`. No work was at risk because everything was committed and pushed, which is the reason to push continuously rather than batch.

## T-01KYYYJ506EGAB6BSVWXW2ZM23 T1036 PR 703 assembled on top of the merged PRs; perfscan unioned
kind: task
state: draft
created: 2026-08-01

PR 703 assembled and pushed after all 52 open PRs were merged.

WHY IT WAS REBUILT RATHER THAN REBASED. Merging the 621-commit branch produced 13 conflicted files, and in most of them BOTH sides had independently optimized the same hot loops - this branch had a KDA decay column-walk fix and Retention output blocking, main had multi-accumulator dots and a loop interchange for the same functions. Taking either side wholesale silently discards a measured optimization, and nothing in the diff shows which. The merge was aborted and this session's work rebuilt on top of main instead: 47 commits touching a known, mostly disjoint file set.

WHAT THAT SURFACED. PR 651 reintroduced the strided out[j*cols+c] read in LU back-substitution that earlier work had removed, and had no column jam. Both were rebuilt AGAINST its flat row-major factor rather than around it, and the win reproduced larger than before: LUSolve_768x768 -72.36%, 512x512 -66.73%, 128x128 -41.62% (p=0.000, n=10), with the cols==1 remainder path flat as a control. New 512 and 768 cells were added first; the pre-existing 64 and 128 cells cannot resolve either transform.

FILE-LEVEL CHERRY-PICKING LEAKS OLDER WORK, and three separate failures proved it. classic/naivebayes.go carried an earlier jointRow signature change that main's own test does not compile against - fixed by taking main's file and applying only this session's hunk. Two autograd bench files needed helpers that live in older test infrastructure, dragging in a dependency chain; they were dropped. nlp/contrastive_search.go could not come at all: this session's windowing applies to a dotAndNorm that does not exist in main, and the 4-partial reassociation it sits on is forbidden by main's TestMaxContextCosineBitExact. That win is real on this branch and NOT portable without changing a bit-exactness pin.

PERFSCAN UNION. The check sets were near-disjoint - 39 checks main lacked, 1 (PS4013) it had - so a three-way merge produced only two conflicts, both resolved as unions. PS4011's over-firing on per-head attention had been fixed twice, differently: main's #685 suppresses attention markers, this branch suppresses architecture-count trip origins after an audit of all 110 hits found zero true positives. Both guards now run, since either matching means the loop is not a scalar recurrence.

TWO COLLISIONS THAT ONLY EXIST AT MERGE TIME. PS6006 was independently assigned by both sides - main's #697 to cross-backend-dtype-gap, this branch to receiver-scratch-buffer - and the local one renumbered to PS6024. sortedKeys was defined on both sides with different signatures, which Go cannot overload. Neither was preventable from one side: the IDs and names were free in every open PR when each was minted. PERF-ID-COLLISION-001 checks open PRs, which is necessary but not sufficient when two branches mint concurrently and neither is a PR yet.

DELIBERATELY NOT IN THE PR: roughly 570 commits of older branch history containing parallel solutions to problems 653, 658, 664, 689, 690 and 691 also solved. Deciding whose version wins is per-file judgment on the user's own parallel work, not something to guess at.

## T-01KYYZSHYCEV2RDRKT0CE1BHNF T1038 main is red: 10 nn fused-parity failures, 4 from the merge batch
kind: task
state: draft
created: 2026-08-01

MAIN IS RED: 10 failing tests in nn, all of the fused-path-versus-dispatch bit-exactness family. Found while validating unrelated work; reported here because it is not visible from any single PR.

SPLIT OF BLAME, measured rather than assumed by testing three commits.
At 575e2558, the main tip BEFORE any of the 52 merges, SIX were already failing: TestDeltaNetFusedBitExactVsDispatch, TestEMAUpdateBitIdenticalToSlowPath, TestGLAFusedBitExactVsDispatch, TestHGRNSeqFusedBitExactVsDispatch, TestRGLRUSeqFusedBitExactVsDispatch, TestTitansLinearFusedBitExactVsDispatch. Those predate this campaign entirely.
The merge batch added FOUR: TestKANFusedBitExactVsDispatch (#661), TestTPAForwardFusedBitExactVsDispatch (#699), TestMemForwardFusedBitExactVsDispatch (#700), TestMTAHeadConvFusedBitExactVsDispatch (#701).
PR 703 added none: the failure set at da813ab8, immediately before it, is identical to the set after.

THE MECHANISM, and it is the transferable part. The TPA failure is a ONE-ULP divergence - fused 0.14259979878693926 against dispatch 0.14259979878693924 - and the test FAILS AT ITS OWN MERGE COMMIT. It was never green on the main it landed on. Each of these four PRs added a fused path plus a test pinning it bit-exact against the dispatch path, was branched from an older main, and passed CI there. By the time it merged, main had moved by dozens of PRs including ones that altered the dispatch side's arithmetic. A squash merge does not re-run CI against the new base, so a PR whose checks are green can still land red.

This is not a conflict and no merge tool would flag it: both sides compile, the diff is clean, and only a bit-exactness assertion notices. It is specific to pins that compare two code paths against each other rather than against a fixed golden, because either side moving breaks them.

WHAT WOULD HAVE CAUGHT IT: requiring each PR to be rebased onto current main and re-run before merge, rather than merging on the checks it earned against its own base. That is expensive at 52 PRs but the alternative is what happened. A cheaper approximation is to run the fused-parity family once after the batch and bisect what it reports, which is how these four were isolated.

NOT FIXED HERE. Four fused paths need re-deriving against the current dispatch arithmetic, and six older ones were already broken. All ten sit in nn, the user's lane, and each needs its own numerical judgment about which side is now correct.
