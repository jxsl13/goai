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
