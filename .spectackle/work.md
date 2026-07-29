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
