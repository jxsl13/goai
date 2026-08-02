---
schema: v1
prefix: A
---

## A-RESTRUCTURED-SCAN-IS-GATED-BY-THE-OLD-IMPLEMENTATION-001
WHEN a change restructures how a loop walks its data (tiling, blocking, batching a per-item scan), the implementer SHALL leaves the superseded per-item routine in place untouched and asserts the new path equals a loop over it, at an item count that is not a multiple of the block width and large enough for more than one block per worker; a GOMAXPROCS(1)-versus-N parity test does NOT satisfy this, since both arms run the new path and every restructuring bug that is not a data race passes it; the gate is proven by 4 mutations reddening it: block scratch not reset between blocks, the wrong item mapped to an output slot, a dropped partial block, and a removed clamp the old routine applied.
