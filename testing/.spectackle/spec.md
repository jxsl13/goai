---
schema: v1
prefix: A
---

## A-RESTRUCTURED-SCAN-IS-GATED-BY-THE-OLD-IMPLEMENTATION-001
WHEN a change restructures how a loop walks its data (tiling, blocking, batching a per-item scan), the implementer SHALL leaves the superseded per-item routine untouched and asserts the new path equals a loop over it; a GOMAXPROCS(1)-versus-N parity test does NOT satisfy this, because both arms run the new path.

## A-BLOCKED-SCAN-ORACLE-EXERCISES-PARTIAL-BLOCKS-001
WHEN gating a blocked or tiled scan against its per-item oracle, the implementer SHALL picks an item count that is not a multiple of the block width and forces more than one block per worker, and reddens the gate with 4 mutations: scratch not reset between blocks, wrong item mapped to an output slot, dropped partial block, removed clamp.

## A-MUTATION-THAT-REDDENS-NOTHING-IS-CHECKED-FOR-HAVING-RUN-001
WHEN a mutation of load-bearing code leaves the suite green, the implementer SHALL confirms the intended test ran under the selector used before concluding anything; a -run pattern that misses a test name reads identical to a passing mutation, and 3 of this rounds first mutations were silent for that reason or for not compiling.

## A-BIT-EXACT-GOLDEN-SKIPS-UNDER-THE-RACE-BUILD-001
WHEN a test asserts exact floating-point bits against a recorded digest, the implementer SHALL skips it when the race detector is on, because the race build inhibits optimizations the compiler may otherwise make and changes the result; the MLA backward digests differently under -race on the pre-change implementation identically, so 2 of 2 goldens would have failed for the build mode rather than the change.

## A-GOLDEN-ON-A-ROUNDED-OUTPUT-CANNOT-PIN-SUMMATION-ORDER-001
WHEN gating a reassociation-sensitive change whose result is stored at lower precision than it is accumulated, the implementer SHALL pins the order on the higher-precision arm and treats the rounded arm as a value check only; reversing the fold order in the f32 MLA arm left its golden green because a float64 accumulator stored back as float32 rounds the difference away, while the identical F64 code reddens.

## THE-NLP-RACE-SUITE-EXCEEDS-40-MINUTES-001
IF go test ./nlp under the race detector reaches its timeout, THEN the implementer SHALL not read it as a regression signal: it ran past both the 10-minute default and an explicit 40-minute limit on the M2 Pro host with no assertion failure, because the package runs many 500-token generate tests and the race build multiplies them; gate instead on the package owning the edit (backend/cpu passes under -race in 113 seconds) plus the full non-race suite.

## A-RANKING-ORACLE-NEEDS-AN-INDEX-DEPENDENT-MUTATION-001
WHEN mutating a score to test a gate whose observable is a selection or ranking, the implementer SHALL makes the perturbation depend on the ranked dimension and large enough to reorder; a uniform scale or offset of one items scores changes no ranking at all, and 3 mutations designed that way read as green before the fourth reddened 4 tests.

## A-GROUP-UNIFORMITY-TEST-STATES-WHAT-IT-GUARDS-001
WHEN a grouped kernel checks its group for uniformity, the implementer SHALL records whether that check guards correctness or only speed; dropping 1 of the 4 mask checks in the masked attention kernel leaves the suite green, so it exists to skip wasted work.

## A-DTYPE-BRANCHED-KERNEL-NEEDS-A-CELL-PER-BRANCH-001
IF a kernel branches on dtype and only one branch has a benchmark, THEN the implementer SHALL not read that cell as evidence about the other branch: the flash attention f32 path was already unrolled and the f64 path was not, so the f64 change read as 2.1 percent noise against the f32 cell and 36 percent against an f64 one added for it.

## A-TEST-FILE-NAME-MUST-NOT-END-IN-A-GOARCH-OR-GOOS-WORD-001
IF a Go file name ends with an underscore-separated architecture or OS word before _test.go, THEN the implementer SHALL rename it: Go applies an implicit build constraint, so ps3054_asymmetric_dtype_arm_test.go landed in IgnoredGoFiles on arm64 and all 4 of its fixtures plus 3 mutations of the check read as green without ever running.
