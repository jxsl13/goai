---
schema: v1
prefix: A
---

## A-REPEATED-SCAN-IS-BANDWIDTH-BOUND-TILE-IT-001
WHEN per-item work re-reads a shared collection that the item loop does not index, the optimizer SHALL treats the loop as bandwidth-bound and tiles the item loop rather than tuning the arithmetic; on the measured site 4 accumulators bought 7 percent while a tile of 16 bought 24.
