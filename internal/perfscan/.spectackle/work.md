---
schema: v1
---

## T-01KYJQ3PGEEFSAKZXXRNP42W5Q Broaden PS1001 — it misses per-element AtF64/SetF64 outside Numel/Unravel loops
kind: task
state: draft
created: 2026-07-27
targets: internal/perfscan/perfscan.go, internal/perfscan/perfscan_test.go

DETECTOR GAP, now diagnosed on a CORRECT baseline and quantified. Prior rounds of this task were measured without perfscan's config, which silently disables the domain checks — that contamination is fixed (see the separate task and commit 5854d2b, which now warns) and every number below was taken WITH -config internal/perfscan/perfscan.json.

BASELINE, config loaded: PS1001 reports 44 findings tree-wide. It does NOT report nlp/kvevict.go, nlp/quant_llama_decode.go, rl/continuous.go or llamagpu/t5_decoder.go.

CAUSE, instrumented at the kvevict SetF64 call site with config loaded:
  name=SetF64  acc=true  hasFlat=false  perElem=false
So the accessor is recognized and no fast-path suppression applies. THE LOOP CLASSIFIER IS THE BLOCKER — perElemLoop is false for `for j := range d` where `d := t.Shape()[1]`, because perElemLoop is set only from a configured element-count method (Numel) or a directly-present Unravel. A shape dimension is neither.

THE OBVIOUS FIX WORKS BUT IS FAR TOO BROAD — this is the finding, and it is why nothing was committed. Adding idents assigned from an IndexExpr over a CallExpr to numelIdents does make kvevict.go:119 report. It also takes the tree-wide count from 44 to 266, a six-fold increase. That predicate matches ANY `x := someCall()[i]`, not just a shape dimension, so most of the 222 new hits are presumed false positives. Unshippable without narrowing; reverted rather than left in.

NARROWING TO TRY, in order: (a) require the call to be a CONFIGURED shape method, adding a shapeMethods list to perfscan.json alongside elementCountMethods — this keeps the rule domain-configured like its siblings and is the smallest honest change; (b) additionally require the ident to be used as a loop bound rather than merely assigned; (c) measure the finding count after each and stop when the delta over 44 is small enough to triage by hand.

THE OTHER SHAPES REMAIN UNSOLVED. The four sites do not share one bound form: a shape-call index (kvevict), a struct-field selector `d := m.Config.Dim` (quant_llama_decode), and len() plus range-over-slice (contMat). (a) above addresses only the first. Each needs its own predicate and its own count check — do not assume one widening covers them.

METHOD NOTE, earned across five rounds on this rule: every intermediate conclusion here was wrong at least once — the bound predicate, then hasFlat, then 'the patch does not populate', then the perElem reading itself (a config artifact). What finally held was measuring one term at a time with the tool configured as CI configures it. Instrument, and check the finding count delta before believing a widening.

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

## T-01KYJW0WF7F8PRB1ZVDK6XEZ4F Perfscan reported zero findings for domain checks running with no vocabulary
kind: task
state: draft
created: 2026-07-27

Perfscan silently runs its four DOMAIN checks (PS1001, PS1002, PS2001, PS4002) with an EMPTY vocabulary when no config is found, printing a clean 'no candidate anti-patterns found' that reads as 'no instances'.

CAUSE: the vocabulary lives in internal/perfscan/perfscan.json, and config discovery walks UPWARD from the working directory. The file sits INSIDE internal/perfscan/, so an invocation from the repo root never finds it. make perfscan passes it via -config; ad hoc invocations do not.

MEASURED on this tree:
  go run ./internal/perfscan -checks PS1001 ./...                                          ->  0 findings
  go run ./internal/perfscan -config internal/perfscan/perfscan.json -checks PS1001 ./...  -> 44 findings

COST, and the reason this is filed rather than shrugged off: it manufactured a multi-round investigation into PS1001 as a broken detector. It is not broken. With an empty accessor set the site can never be reported regardless of loop shape, and an instrumented reading that appeared to indict the loop classifier (perElem=false) was itself an artifact of the same emptiness. Two rounds of otherwise-sound diagnosis chased a configuration default. This is the same false-assurance failure already recorded in FMT-004 for a fuzz target that never reached its parser: the tool answered confidently about a question it was not equipped to evaluate.

FIXED (commit 5854d2b): perfscan now warns to stderr, naming the starved checks and the remedy, whenever a domain check is enabled with no vocabulary loaded. Silent when a config is present, so make perfscan and CI are unaffected. Fixture suite unchanged and green.

DELIBERATELY A WARNING, NOT A REFUSAL: perfscan is repo-agnostic by design and the language-shape checks are genuinely useful with no config at all, so failing hard would break legitimate stdlib-only scanning. The warning names what is inert and why, which is enough to stop the misreading.

FOLLOW-UP worth considering separately, not done here: move perfscan.json to the repo root so upward discovery finds it, or have perfscan also look next to its own package. Either would remove the trap rather than annotate it. Both are behavioral changes to config resolution and deserve their own decision.
