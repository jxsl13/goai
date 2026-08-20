---
schema: v1
prefix: HOST
---

## HOST-RESIDENT-EMBED-BACKWARD-002
WHEN receives a valid host-resident F32 table, index vector, and upstream gradient, the Vulkan OpEmbedBackward SHALL execute exactly 0 Vulkan submissions and return the deterministic reference-order scatter-add gradient.

## MEASURED-VULKAN-BIAS-GRAD-ROUTE-001 {applies: go:vulkan.addBiasBackwardF32}
WHEN a synchronous host-resident F32 bias gradient is requested, the Vulkan add-bias backward SHALL route through CPU only where 3 count-7 campaigns each prove at least 1.10x median speedup, and preserve Vulkan elsewhere.

Rationale: The execution side must follow measured end-to-end transfer cost rather than nominal backend affinity.
