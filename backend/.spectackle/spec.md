---
schema: v1
---

## intent
- T-01M0FQ2Z5MFQ1V30BDS7GGRDTF Gate deterministic host Vulkan embedding backward: Implement a typed F32 host scatter in Vulkan embedBackwardF32, preserve reference per-add rounding and repeated-index accumulation, preserve the old SPIR-V atomic path as a same-binary control, run the five frozen M2 MoltenVK shapes across three independent campaigns, validate the full Vulkan and repository suites, update perfscan#771, and use a dedicated PR merged only after every CI check passes [body truncated at tombstone retention cap]
- P-01M0FQ0RWQE8PT2GT7FFP4WKBE Route host-resident Vulkan embedding backward deterministically: Context: merged Metal evidence P-01M0FNKC7DEJZ proved that host-resident synchronous embedding backward loses to a typed host scatter. Vulkan currently repeats the same upload of indices, upstream gradient, and zero table, atomic dispatch, synchronization, and full-table download. ADR-01M0FQ1AP8FBC conditionally selects the typed host route while persistent device residency remains a separate grap [body truncated at tombstone retention cap]
