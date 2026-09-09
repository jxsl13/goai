# Prospective control allocation-site investigation

Status: read-only research and plan review; no site diagnostic has been
implemented or executed. Spectackle research `R-01M23VWNB5FRP` follows the
[complete fixed-count diagnostic](../m2-control-alloc-diagnostic-20260909/README.md).
Both original GELU candidate rejections remain unchanged.

`research.txt` preserves the initial source-backed proposal, not an approved
measurement protocol. It traces unchanged control/output/pool paths and pinned
Go 1.27.1 profiling semantics. A separate rate-1 experiment would be perturbed
diagnostic evidence, not interchangeable with unprofiled qualification.

Root review identified two questions for fresh independent plan review:

- Public `runtime.MemProfileRecord` has counts and bytes but no ObjectSize.
  With `inuseZero=true`, this SDK's internal enumeration can return zero-active
  buckets. Size derivation needs a zero/unknown policy; 32-PC truncation also
  requires aggregation of visible-key collisions without discarding raw rows.
- Cross-binary matching by absolute source line can misidentify the same
  allocation after preceding source insertions. Raw locations must be retained,
  but a comparison key must avoid manufacturing differences from line shifts
  and clearly disclose any coarser function-stack aggregation.

These are prospective instrumentation design concerns, not observed causes of
the completed Softplus differences. No profile, site attribution, gate revision,
noise subtraction, or runtime promotion is claimed here.
