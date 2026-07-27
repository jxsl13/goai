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

SECOND CONFIRMED DETECTOR BUG this round, with the root cause isolated in the perfscan source and verified by running the tool.

ROOT CAUSE, two independent gaps:
1. internal/perfscan/perfscan.go:470-482 integerKeyType starts with id, ok := e.(*ast.Ident); if !ok { return false }. A key spelled backend.Op is an *ast.SelectorExpr, so it returns false on the FIRST LINE. A locally named type (map[Op]VJP, an *ast.Ident named Op) also fails, one line later, at the switch over builtin names. Every named integer type — the entire op-dispatch vocabulary of this repo — is invisible.
2. intKeyMapNames (perfscan.go:488-536) harvests from *ast.Field, *ast.ValueSpec and *ast.AssignStmt. For a package-level var vjps = map[backend.Op]VJP{} the ValueSpec has Type == nil (the type lives on the RHS composite literal), so add(x.Type, ...) is a no-op. The ValueSpec arm never inspects x.Values, unlike the AssignStmt arm which does handle a CompositeLit.

CONFIRMED EMPIRICALLY: go run ./internal/perfscan -checks PS3003 ./autograd/... reports PS3003 four times, ALL on firstPos (autograd/vjp_einsum.go:71,97,115,124 — a plain map[int]int), and ZERO times on autograd/autograd.go:176 rule, ok := vjps[n.op], whose registry is declared at autograd/vjp.go:19 as map[backend.Op]VJP. The multi-output twin at autograd.go:218 is equally invisible.

WHY THIS IS SYSTEMIC, not a one-off miss: backend.Op is a dense iota enum (backend/op.go:7, OpInvalid = 0 through numOps at :158) and the repo ALREADY uses the array-table idiom for exactly this key — var opAttrsSpec = [numOps]attrsSpec{...} at backend/attrs.go:632, with NumOps exported at :589. So the codebase both suffers the pattern and already knows the fix, while the detector meant to find it cannot see any of it. This blinds PS3003 to every enum-keyed dispatch table in a library whose entire architecture is enum-keyed dispatch.

FIX: teach integerKeyType to accept *ast.SelectorExpr and non-builtin *ast.Ident keys as CANDIDATE integer types, gated on go/types — types.Info.TypeOf(mt.Key).Underlying() being a *types.Basic whose Info() has types.IsInteger set. Extend the ValueSpec arm of intKeyMapNames to inspect x.Values[i] composite literals when x.Type == nil. Then RE-SCAN THE WHOLE REPO and record the full new finding set: this will also surface backend.Context.opBackends map[Op]Name (backend/backend.go:60) and any kernel registry keyed the same way, and each newly flagged site must be classified as a real candidate or a false positive individually rather than assumed.

VALIDATION GATE: a detector-correctness task, so the gate is fixtures plus a repo sweep, NOT a benchmark. Add autograd.go:176 and the map[backend.Op]VJP declaration form as POSITIVE fixtures; add a negative fixture with a sparse or non-integer named key so the rule is proven non-vacuous in both directions; confirm the four existing firstPos positives still fire.

SEPARATE, OPTIONAL FOLLOW-UP — do NOT bundle it: converting vjps/vjpsMulti to [backend.NumOps]VJP arrays saves roughly 8-12 ns/node (a mapaccess2_fast64 against a bounds-checked array load), which on a 2000-node transformer backward is about 20 us/step, i.e. 1-2% — honestly small. If taken, RegisterVJP keeps its duplicate panic by testing != nil and the lookup becomes a range check plus a nil test, preserving the same !ok semantics and the same error text at autograd.go:178. It needs its own benchmark (see the backward-walk depth benchmark in the tape-accumulate task) and is bit-identical since no arithmetic is touched.

RELATION TO THE PS1001 TASK: same class of failure, different rule — a detector whose predicate is narrower than the anti-pattern it names, producing clean scans that are read as "no instances". Two such bugs found in two sweeps suggests the fixture suites test that rules FIRE, but not that they fire on the shapes that actually occur in this codebase. Worth stating in the closing note.

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
