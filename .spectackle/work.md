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

## T-01KYJX48GFEHZBYPPGBQA9Z4WT tensor gatherCast: hoist innermost axis out of the strided-gather odometer
kind: task
state: draft
created: 2026-07-27

TARGET: gatherCast[S, D] in tensor/tensor.go (odometer at the loop containing idx[d]++ near line 210). It walks an N-D strided view element by element, paying a full nd-deep odometer tick per element:
    for pos := range dst { dst[pos] = D(src[off]); for d := nd-1; d >= 0; d-- { ...tick... } }

TRANSFORM (identical in shape to the two already shipped): hoist the innermost axis out of the odometer. Its stride is constant across a run, so a run is a straight walk `s := off; for p := range run { run[p] = D(src[s]); s += sInner }`, with the odometer ticking once per run instead of once per element. gatherBlocked2D in the same file already uses exactly this inner-loop trick, so the idiom is established locally. Precedent: backend/ref/broadcast.go measured 4.49x, backend/cpu/elementwise.go measured 5.29x (commit 5bfa77b), both bit-identical.

Bit-identity is by construction: pure traversal reordering, no arithmetic on values, every dst position still reads the same src offset. The innermost axis's net contribution to off over a full run is inner*sInner - sInner*inner = 0, leaving the outer tick unchanged. The D(src[s]) conversion is per element in both forms, so the cast sequence is untouched.

LEVERAGE IS UNESTABLISHED AND MUST BE MEASURED BEFORE ANY WORK - this is the residual path, not the common one. tensor.go already specializes the two shapes that matter most: gatherRows handles innermost-stride-1 views (what Slice on a non-last axis produces) and gatherBlocked2D handles rank-2 transposes. gatherCast only receives what neither covers - rank>=3 views whose innermost stride is neither 1 nor a rank-2 transpose, for example a permuted attention tensor. That residual may be rare enough that the win never appears in a real workload.

MANDATORY SEQUENCE, in this order (the failure mode this guards against has already occurred twice this line of work):
1. Establish that a candidate benchmark ENTERS gatherCast, by temporarily panicking inside it and confirming the panic fires. Do not read any timing number before this passes. An existing benchmark whose name matches "contiguous" or "permute" is not evidence - gatherRows and gatherBlocked2D shadow this function for the obvious shapes.
2. If no existing benchmark enters it, write one over a rank-3+ permuted view whose innermost stride is neither 1 nor a rank-2 transpose, and re-run step 1 against it.
3. Only then run the interleaved A/B: file-copy toggle, same host and session, three A-B alternations, medians, plus an UNAFFECTED control benchmark that must not move. If the control drifts more than the candidate delta, the result is noise - discard it.
4. Bit-identity: byte-compare raw output bits (math.Float64bits, no tolerance) across the S/D instantiations, not just float64/float64.
5. Non-vacuity by MUTATION, per arm, and scope the mutation run to the test that claims the coverage. A test whose case name says which arm it exercises is not proof it does - that exact overclaim was caught in the cpu work, where the case named for the fill arm never reached it.

REJECT AND REVERT if the control drift swamps the delta, or if no benchmark can be made to enter the path - a transform with no reachable hot caller is not worth the risk to a function this central.

PERFSCAN COVERAGE (standing requirement): this site is already machine-findable - PS1001 reports tensor/tensor.go in the current scan, and the per-element-odometer class is what PS4004 and PS1001 exist to surface. No new rule is required for this optimization. If the implementation reveals a shape neither rule catches, add the rule before shipping the code.

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
