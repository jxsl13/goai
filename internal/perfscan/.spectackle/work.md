---
schema: v1
---

## T-01KYJQ3PGEEFSAKZXXRNP42W5Q Broaden PS1001 — it misses per-element AtF64/SetF64 outside Numel/Unravel loops
kind: task
state: draft
created: 2026-07-27
targets: internal/perfscan/perfscan.go, internal/perfscan/perfscan_test.go

DETECTOR BUG. PS1001 does not fire on per-element AtF64/SetF64 loops its name implies it covers. FIVE confirmed sites: nlp/quant_llama_decode.go:25-27, nlp/kvevict.go:115-123, llamagpu/t5_decoder.go:226, rl/continuous.go:245-254, and nlp/streaming.go keepSinkRecent (since fixed under its own task, but never flagged).

INSTRUMENTED DIAGNOSIS — the blocker is narrowed to ONE of three conditions, so the next attempt need not guess. The predicate is perElem AND accessors[name] AND NOT hasFlat. Printing all three for nlp/kvevict.go gives, at the SetF64 call site:

  kvevict.go:120:4  name=SetF64  perElem=false  hasFlat=false

So the accessor IS recognized, and hasFlat is NOT suppressing it — a hypothesis worth discarding explicitly, since it was the leading suspect. THE LOOP CLASSIFICATION IS THE BLOCKER: perElem is false for a range over an ident bound to a shape dimension.

WHAT THAT MEANS FOR THE FIX. perElemLoop is set from isNumelRange/isNumelForCond (bound derived from a configured element-count method) or directlyHasUnravel. A range over a shape-dimension ident satisfies neither. An earlier attempt added idents assigned from an IndexExpr over a CallExpr (the shape-call-index form) to numelIdents; it did NOT make the site fire and was reverted rather than left in as unvalidated speculation. Now that perElem is confirmed as the failing term, that direction is right but something in it did not take — instrument numelIdents itself next, printing whether the dimension ident is present when GatherRows is scanned, before changing the predicate again.

THE FIVE SITES DO NOT SHARE ONE BOUND SHAPE. Three distinct ones: a shape-call index (kvevict), a struct-field selector (quant_llama_decode), and len() plus range-over-slice (contMat). A predicate widened for one still misses the others — fix and verify each shape separately.

PRECEDENT WORTH FOLLOWING: the sibling PS3003 bug was closed this session and its final cause was STRUCTURAL, not a predicate — a file-scoped fact that had to become package-scoped. Three reasoned fixes there each left the reproduction silent; one instrumented print located it in a single run. Expect the same shape of surprise here, and re-check the reproduction after every change rather than trusting the reasoning.

VALIDATION GATE: fixtures plus a repo sweep, NOT a benchmark. Add all five sites as positive fixtures, plus a negative where the accessor sits in a genuine dtype fallback branch, so the rule is proven non-vacuous both ways. Then re-scan and classify every new finding per site.

## T-01KYJR34RJE7HSS68635PKJYZ2 PS3003 is blind to named integer map keys — it cannot see any enum-keyed dispatch table
kind: task
state: done
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
