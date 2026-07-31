---
schema: v1
prefix: SLOT
---

## SLOT-IS-PER-RUNNER-NOT-PER-UNIT-001
IF a work-partitioning primitive hands callers an index for per-partition scratch, THEN the implementing agent SHALL define it as a per-RUNNER slot bounded by the worker count, not a per-unit index.

Rationale: The property a caller scratch buffer depends on is that no two CONCURRENT runners share a slot. Under a static one-chunk-per-worker split that is indistinguishable from handing each unit a unique index, so the weaker definition survives untested. It stops being equivalent the moment units are claimed rather than dealt, which is what balances a heterogeneous CPU: one runner takes several units and is called several times, safely, with its own stable slot. A test asserting each index is used at most ONCE then fails on correct code and blocks the optimization. Assert occupancy instead: mark the slot on entry, clear it on exit, and fail if a runner finds its own slot marked. That test must also HOLD the slot long enough for runners to overlap. Verified by mutation: handing every runner slot 0 passes with a trivial body, because the caller drains most units before a worker receives its task, and fails once the body spins for 200 microseconds.
