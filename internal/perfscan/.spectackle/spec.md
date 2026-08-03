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
