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
