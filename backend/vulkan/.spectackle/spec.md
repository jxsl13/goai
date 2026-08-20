---
schema: v1
prefix: HOST
---

## HOST-RESIDENT-EMBED-BACKWARD-002
WHEN receives a valid host-resident F32 table, index vector, and upstream gradient, the Vulkan OpEmbedBackward SHALL execute exactly 0 Vulkan submissions and return the deterministic reference-order scatter-add gradient.
