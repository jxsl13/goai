---
schema: v1
---

## T-01KYJQ3PGEEFSAKZXXRNP42W5Q Broaden PS1001 — it misses per-element AtF64/SetF64 outside Numel/Unravel loops
kind: task
state: draft
created: 2026-07-27
targets: internal/perfscan/perfscan.go, internal/perfscan/perfscan_test.go

DETECTOR BUG — but the evidence base was contaminated and is now corrected. Read this section before acting on anything below it.

THE CORRECTION. Every earlier measurement in this task was taken by invoking perfscan WITHOUT its config, which silently disables the four DOMAIN checks. internal/perfscan/perfscan.json supplies elementAccessors (AtF64, SetF64) and friends, and it is discovered by walking UPWARD from the working directory — so running from the repo root never finds it, because the file sits inside internal/perfscan/. make perfscan passes it via -config; ad hoc invocations do not. Measured:
  go run ./internal/perfscan -checks PS1001 ./...                                   ->  0 findings
  go run ./internal/perfscan -config internal/perfscan/perfscan.json -checks PS1001 ./...  -> 44 findings
So PS1001 is NOT globally broken. The instrumented reading that pointed at the loop classifier (perElem=false) was itself an artifact: with no config the accessor set is empty, so ns.accessors[AtF64] is false and the site can never be reported regardless of loop shape. Two rounds of diagnosis chased that.

WHAT SURVIVES THE CORRECTION. Re-run WITH the config, the four originally cited sites are STILL not reported:
  nlp/kvevict.go, nlp/quant_llama_decode.go, rl/continuous.go, llamagpu/t5_decoder.go  -> 0 each
So a real gap remains, but it is much narrower than documented and its cause is unknown again — the perElem evidence is void. Start over from a correct baseline.

MANDATORY FIRST STEP for whoever continues: always pass -config internal/perfscan/perfscan.json (or run make perfscan). Any finding count taken without it is meaningless for the domain checks.

WHAT IS STILL KNOWN GOOD, measured with the config absent but independent of it: the five sites are genuine per-element AtF64/SetF64 loops on hot paths, and they share no single bound shape — a shape-call index (kvevict), a struct-field selector (quant_llama_decode), and len() plus range-over-slice (contMat). Whatever the cause turns out to be, a predicate widened for one shape will still miss the others.

A CANDIDATE, NOT A CONCLUSION: adding idents assigned from an IndexExpr over a CallExpr to numelIdents does populate correctly (verified: numelIdents=map[d:true] for GatherRows) and does flip the inner loop to perElem=true (verified: kvevict.go:119:3 perElem=true). It was reverted because it could not be shown to change any reported finding. Re-evaluate it against a config-loaded baseline before adopting or discarding.

WORTH FIXING SEPARATELY, and arguably the more valuable defect: perfscan runs the domain checks SILENTLY with an empty vocabulary when no config is found. That is the same false-assurance failure this rule family keeps producing — a clean scan that reads as no instances. It should warn, or refuse, when a domain check is requested with no accessor vocabulary loaded. File it as its own task.

VALIDATION GATE: fixtures plus a config-loaded repo sweep, NOT a benchmark. All five sites as positive fixtures, plus a negative where the accessor sits in a genuine dtype fallback branch. Then classify every finding in the 44 per site.

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
