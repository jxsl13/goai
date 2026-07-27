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
