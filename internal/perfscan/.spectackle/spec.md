---
schema: v1
prefix: PERF
---

## PERF-ID-COLLISION-001
WHEN a new check ID is minted, the perfscan check registry SHALL reject any ID not verified free on main and in every open PR touching the registry, via `gh pr diff` per PR.

Rationale: Check IDs are documented as stable and are the handle used by //perfscan:ignore directives and -checks= selection. Two branches minting the same next-free ID collide silently: both build, both pass their own suite, and whichever merges second leaves one ID carrying two meanings, so an ignore directive written for one check suppresses the other. Observed when PR #394 (PS4004 scalar-copy-loop, PS6001 unverified-dual-path) and PRs #412/#413 independently minted PS4004 and PS6001 against a main that had neither; resolved by renumbering #412 to PS4007 and #413 to PS3004, both free in either merge order.

## PERF-SUPPRESS-INERT-001
WHEN a //perfscan:ignore directive is written, the a perfscan suppression SHALL re-run the scan and confirm the finding is gone; a directive that fails to apply is silently inert and reads as accepted.

## PROC-PROFILE-PARKED-001
IF a parallel program profiles as mostly runtime synchronization, THEN the reader SHALL measure the GOMAXPROCS 1/4/12 curve before calling it overhead; parked time is attributed but free, and a spin-before-park replacement measured 2-4% slower.

## PERF-CHECK-ADVICE-MUST-TRANSFER-001
WHEN a perfscan check states a remedy generalized from one measured site, the implementing agent SHALL re-measure that remedy on a site with the opposite memory-access shape before stating it unconditionally.

Rationale: PS1007 was generalized from a sparse P·V where the input was read d-STRIDED, and its remedy — strip-mine the inner loop by 4 with the outer loop innermost — was stated as the fix. Applied to linalg QR, where the input is already read as a CONTIGUOUS row slice, that remedy wins only at small sizes and then evaporates: LstsqMat -2.56% at n=64, -1.92% at 256, -0.52% at 512, indistinguishable from base at 768, because it trades one contiguous pass over the submatrix for n/4 strided passes whose cost grows with the outer trip count. The remedy that wins on the contiguous shape is the opposite move — unroll the OUTER loop by 2 with separate accumulating adds, keeping the inner pass contiguous while the accumulator lives in a register across the pair: -2.31% to -3.16% at every size, geomean -2.69%. Both forms are bit-identical, so only measurement could separate them. The check now carries both remedies plus the discriminator. The failure mode is specific: a check derived from one site encodes that site accidental access shape as if it were the pattern, and the only way to find the missing case is to run its own advice on a site that differs in exactly that respect.
