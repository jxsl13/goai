---
schema: v1
prefix: PERF
---

## PERF-ID-COLLISION-001
WHEN a new check is added, the perfscan check registry SHALL its ID MUST be verified free on main AND in every open PR that touches the registry — grep the candidate ID out of `gh pr diff <n>` for each open PR before minting it.

Rationale: Check IDs are documented as stable and are the handle used by //perfscan:ignore directives and -checks= selection. Two branches minting the same next-free ID collide silently: both build, both pass their own suite, and whichever merges second leaves one ID carrying two meanings, so an ignore directive written for one check suppresses the other. Observed when PR #394 (PS4004 scalar-copy-loop, PS6001 unverified-dual-path) and PRs #412/#413 independently minted PS4004 and PS6001 against a main that had neither; resolved by renumbering #412 to PS4007 and #413 to PS3004, both free in either merge order.
