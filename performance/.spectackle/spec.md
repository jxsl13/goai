---
schema: v1
prefix: A
---

## A-REPEATED-SCAN-IS-BANDWIDTH-BOUND-TILE-IT-001
WHEN per-item work re-reads a shared collection that the item loop does not index, the optimizer SHALL treats the loop as bandwidth-bound and tiles the item loop rather than tuning the arithmetic; on the measured site 4 accumulators bought 7 percent while a tile of 16 bought 24.

## A-REUSED-STAGING-WINDOW-CHECKS-CONSUMER-ACCUMULATION-001
WHEN replacing a whole-tensor staging buffer with a reused per-chunk window, the optimizer SHALL establishes whether the consuming kernel stores or ACCUMULATES into it, and clears the window between chunks when it accumulates; the whole-tensor form wrote each row slot once, so an accumulating consumer is indistinguishable from a store there.
