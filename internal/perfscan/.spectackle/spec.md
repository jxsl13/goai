---
schema: v1
prefix: PERF
---

## PERF-ID-COLLISION-001
WHEN a new check ID is minted, the perfscan check registry SHALL reject any ID not verified free on main and in every open PR touching the registry, via `gh pr diff` per PR.

Rationale: Check IDs are documented as stable and are the handle used by //perfscan:ignore directives and -checks= selection. Two branches minting the same next-free ID collide silently: both build, both pass their own suite, and whichever merges second leaves one ID carrying two meanings, so an ignore directive written for one check suppresses the other. Observed when PR #394 (PS4004 scalar-copy-loop, PS6001 unverified-dual-path) and PRs #412/#413 independently minted PS4004 and PS6001 against a main that had neither; resolved by renumbering #412 to PS4007 and #413 to PS3004, both free in either merge order.

## PERF-SUPPRESS-INERT-001
WHEN a //perfscan:ignore directive is written, the a perfscan suppression SHALL re-run the scan and confirm the finding is gone; a directive that fails to apply is silently inert and reads as accepted.

## intent
- R-01KZ11ASDMFQVS5D6FYJS94MBV Invariant-nest check: three attempts, all withheld — the motivating transform is loop DISTRIBUTION, not hoisting: No action: three predicates drafted and all withheld, with each failure diagnosed. The conclusion is that the motivating transform is loop DISTRIBUTION rather than hoisting — the invariant work is a statement prefix inside a loop that is itself dependent — so no loop-level AST predicate can see it. Recorded to stop a fourth attempt from repeating the first three.

## A-NEW-PARALLELISM-CHECK-REPORTS-ITS-OWN-CONVERSIONS-001
WHEN a new check looks for unused parallelism, the its author SHALL run it tree-wide and classify EVERY candidate before shipping, because the shapes an applied conversion leaves behind look exactly like the shape being hunted.

Rationale: PS3063 shipped with 5 candidates and 3 of them were artifacts. Applying the transform leaves a plain duplicated loop beside the gated dispatch - which is PS3040 own advice, since routing small inputs through the callback costs a few percent - so the leftover arm is the same shape; the LU factorization in linalg and the AQLM Gauss-Jordan, both already converted, were reported by their own leftovers. A sequential outer loop that fans out its inner one is PS3040 shape and contains the dispatch directly, so it was reported too. And a closure assigned to a name at the top of a function and invoked only from inside a callback is already parallel while nothing in its syntax says so - the conv2d backward kernel declares col2im that way. One test covers the first two, whether the nest contains a fan-out call, and a second tracks function literals assigned to names that a callback then invokes. 5 candidates down to 2, both plausible. THIS IS THE THIRD CHECK TO NEED THE BELOW-GATE-TWIN GUARD after PS3040 and PS3056, so it is not a quirk of one check: any check that names a transform will, once the transform is applied, find the residue of its own advice. Classify the whole candidate list against the code before believing the count.

## PS3059-DOES-NOT-ENUMERATE-A-FUNCTIONS-NESTS-001
IF PS3059 reports one nest in a function, THEN the reader SHALL read the whole function anyway, because the check has been observed to report the colder of two nests of its own shape and the cause is not yet isolated.

Rationale: The LLM.int8 kernel held two nests of the derived-base shape in one function. The check reported only the first, at line 174 of that file; the second, at 233, was 90.9 percent of its benchmark and 5.0x once banded, and was found by a scaling probe rather than by the tool. Removing the one-finding-per-function limit did NOT surface it - verified by running the current build against the pre-change file, which still reports only 174. Lifted into a fixture verbatim, with a fan-out helper declared beside it, the second nest FIRES, and instrumentation logs every condition as satisfied: outer variable i, depth 2, one indexed write, owned, reached through a derived base. So the shape is right and something about the surrounding file suppresses it. Three rounds of bisection did not isolate what, and the earlier hypothesis that the first-finding limit was the cause is now measured false. Recorded in the checks own comment so the next reader starts here rather than from scratch.

## VERIFY-A-PATCH-LANDED-IN-THE-FUNCTION-YOU-MEANT-001
WHEN a scripted source edit asserts its anchor is unique before replacing, the author SHALL verify afterwards that the edit landed in the intended function, because a uniqueness assert on a short prelude can pass against a match elsewhere in the file.

Rationale: A change to make PS3059 report every nest in a function rather than the first was written, asserted unique, committed, and claimed in a commit message. It never landed. The anchor was a four-line ast.Inspect prelude that seven detectors in the file share, so the assert passed against a different one. Two subsequent rounds then chased why the check still reported only the first nest, and one of them recorded the first-finding limit as a measured NON-cause on the strength of a run whose binary did not contain the change. Instrumenting the real CLI path settled it in one run: the walk logged three nests, accepted the third, and stopped - the signature of the limit still being present. With the edit actually applied and verified by grep, the pre-change LLM.int8 file reports BOTH nests, including the one that was 90.9 percent of its benchmark and 5.0x once banded. Tree-wide PS3059 went 21 to 25, and the same correction applied to PS3063 and PS3065 took them 2 to 3 and 17 to 18. THE CHEAP CHECK IS A GREP FOR THE REMOVED TEXT INSIDE THE TARGET FUNCTION, not a rerun of the tool - the tool agreeing with the old behavior is exactly what a failed edit looks like.
