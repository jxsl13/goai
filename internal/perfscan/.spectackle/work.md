---
schema: v1
---

## T-01KYJQ3PGEEFSAKZXXRNP42W5Q Broaden PS1001 — it misses per-element AtF64/SetF64 outside Numel/Unravel loops
kind: task
state: draft
created: 2026-07-27
targets: internal/perfscan/perfscan.go, internal/perfscan/perfscan_test.go

DETECTOR BUG, confirmed by running the tool. PS1001 does not fire on per-element AtF64/SetF64 loops that its name implies it covers.

CONFIRMED FALSE NEGATIVES (go run ./internal/perfscan -checks PS1001 reports nothing on any of these):
- nlp/quant_llama_decode.go:25-27 QuantLlama.embedOne — the sole surviving per-element embedding lookup; 28 of the package 29 embedOne implementations already call the bulk embedRow.
- nlp/kvevict.go:115-123 GatherRows — row gather where a per-row copy is available.
- llamagpu/t5_decoder.go:226 — per-token decode path, in a package whose decoder.go:3224 already has the bulk embedRow helper.
- rl/continuous.go:245-254 contMat — 3 calls x BatchSize x actDim per SAC.learn, which runs every env step; its sibling rl/rl.go:143-149 already received this fix.
- nlp/streaming.go keepSinkRecent — SINCE FIXED under its own task, but it was never flagged either.

CORRECTED ROOT-CAUSE ANALYSIS. The earlier note in this task said the cause was the Numel/Unravel loop-bound predicate and implied a one-line widening. THAT IS WRONG, and an attempt proved it. Two findings:

(1) The five sites do not share one bound shape. They have THREE: a struct-field selector (d := m.Config.Dim, then range d); len() plus range-over-slice (n, d := len(rows), len(rows[0]), then for i, r := range rows / for j, v := range r); and a shape-call index (d := t.Shape()[1], then range d). A predicate widened for any one of them still misses the others.

(2) Widening the bound predicate is NOT SUFFICIENT. I implemented the shape-call-index case — an ident assigned from IndexExpr whose X is a CallExpr, added to numelIdents so isNumelRange/isNumelForCond classify the loop as per-element — and PS1001 STILL did not fire on nlp/kvevict.go:115, which uses exactly that form. So a second suppression is in play downstream of the loop classification: most likely the hasFlat fast-path check (the function may reach a typed bulk accessor elsewhere and be treated as having a fallback), or the accessor/anyAncestorPerElem attribution. That change was REVERTED rather than left in as unvalidated speculation.

WHAT THIS TASK MUST DO, revised: instrument before patching. Take nlp/kvevict.go:115 as the single reproduction case, determine WHY it is suppressed with the loop correctly classified, and fix that. Only then widen the bound predicate, and widen it to cover all three shapes above rather than one. Add all five sites as positive fixtures plus a negative where the accessor sits in a genuine dtype fallback branch, so the rule is proven non-vacuous in both directions. Finally re-scan the tree and classify every new finding per site.

METHOD LESSON worth carrying: the original analysis inferred the bound shape from a description instead of reading the five sites, and the resulting one-line fix was dead code. Read the sites first.

## T-01KYJR34RJE7HSS68635PKJYZ2 PS3003 is blind to named integer map keys — it cannot see any enum-keyed dispatch table
kind: task
state: draft
created: 2026-07-27
targets: internal/perfscan/perfscan.go, internal/perfscan/perfscan_test.go

ROOT CAUSE FOUND — by instrumentation, after three reasoned fixes each failed. The remaining defect is STRUCTURAL, not a predicate.

THE CAUSE: intKeyMapNames is FILE-SCOPED (it takes a single *ast.File), but a dispatch registry is declared in one file and read in another. vjps is declared at autograd/vjp.go:19; the hot read is at autograd/autograd.go:176. When autograd.go is scanned, intKeyMaps is EMPTY — verified by instrumenting the call site, which printed an empty map. No amount of key-type widening can help, because the map name never enters the set for that file.

Established along the way, so nobody re-checks these: the named-integer-type registry DOES populate correctly (instrumented: intTypeReg[backend][Op]=true, 6 packages), the site IS inside a loop (the reverse walk in runBackward), and the declaration IS the fixed ValueSpec-with-composite-literal shape. Those three are not the problem.

THE FIX: make map-name collection PACKAGE-SCOPED, exactly as the intTypeReg pre-pass already is. Collect intKeyMapNames over all files of a package before scanning any of them, keyed by package, and have scanFunc consult the package entry rather than a per-file map. The pre-pass infrastructure exists — collectIntTypes already walks all parsed files up front — so this is a small extension of it, not new machinery.

Two things to get right while doing it: a package-scoped set makes a same-named local in another file collide, so key on package and prefer a file-local declaration when both exist; and this widening will raise the finding count again, so re-triage rather than assuming the delta is all real.

CORRECTION CARRIED FORWARD (previously overstated, now confirmed): the earlier 4 -> 32 jump came ENTIRELY from the ValueSpec/composite-literal fix and is all plain integer-keyed LOCAL maps (size, pos, children, byFirst, val). The named-type registry contributed zero findings, because every enum-keyed registry in this repo is declared in a different file from its hot read — which is precisely the structural bug above. The two facts explain each other.

STATUS: the ValueSpec fix and the type registry are committed and green (591f81b). The package-scoping fix is the remaining work, and the 28 new findings are still untriaged.

METHOD, earned three times over in this rule family: an isolated cause can be real, necessary, and still not sufficient. Instrument the reproduction before patching — the print that found this took one run, after three reasoned fixes had each left the site silent.

## T-01KYJSBE7HE87A9AWR362JVB3Z Triage the 22 PS4004 findings and tighten the rule if the false-positive rate warrants it
kind: task
state: draft
created: 2026-07-27

FOLLOW-UP to the PS4004 detector added alongside the ref broadcast fix. The rule ships; this task closes the loop on its finding set, which the standing contract requires be classified per site rather than assumed.

WHAT SHIPPED: PS4004 scalar-copy-loop — a counted loop whose only data statement is dst[i] = src[j] between distinct slices, with no arithmetic on the value. Gates: isCountedLoop (three-clause for, or range over a call) excludes rank-sized range-over-container setup loops; conditionalBefore excludes guarded stores, which are filtered scatters rather than movable runs; loopBodyHasCall excludes bodies already reaching for copy() or a helper, while deliberately NOT counting len/cap, which are free index bookkeeping. Positive and negative fixtures live in perfscan_test.go, and the rule is proven against the real case: it fires on exactly the two hot loops of the pre-fix broadcastKernel and is silent on the fixed version.

MEASURED FALSE-POSITIVE HISTORY, recorded because it shaped the gates: the first version reported 30 sites tree-wide. Sampling three found at least two false positives — a guarded scatter in autograd/vjp_reduce.go:197 (the store sits under an if, so no bulk copy can replace it) and backend/cpu/elementwise.go:40, a RANK-sized loop that merely happens to be written three-clause, i.e. exactly the class the counted-loop gate was meant to exclude. Adding conditionalBefore took the set from 30 to 22 while keeping both true positives. A later sample of three (autograd/vjp_ia3.go:38, backend/einsum.go:128, nlp/quant_mixtral_gguf.go:370) looked genuine — dls[j] = acc[j] is a literal copy, and the transpose-style scatter is a real strided-run case.

WHAT THIS TASK MUST DO: walk all 22 sites and classify each as either a real candidate deserving its own task, or a false positive that must be excluded BY CONSTRUCTION rather than by a suppression comment. State the verdict per site. If the false-positive share exceeds roughly a quarter, tighten the rule further before leaving it enabled — a noisy rule trains readers to ignore the scan, which is the exact failure mode already recorded for PS1001 and PS3003 in their own tasks. Candidate further gates, in order of likely value: require the destination index to be affine in the loop variable, since a scatter to a computed permutation cannot be a contiguous run; and require the loop body to contain no other assignment to the destination slice.

EXPLICITLY NOT A BENCHMARK TASK. The gate here is the classified finding set plus the fixture suite, not an A/B — nothing in this task changes runtime behavior.

NOTE ON RANK LOOPS: the counted-loop gate is imperfect by construction. A rank-sized loop written as for d := 0; d < ndo; d++ is indistinguishable from an element loop by shape alone; only the bound provenance separates them, and that is domain knowledge (the Numel vocabulary) rather than a language shape. Deciding whether PS4004 should become a DOMAIN check gated on that vocabulary, as PS1001 is, is part of this triage.
