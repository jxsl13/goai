# Prospective control allocation-site investigation

Status: corrected prospective protocol independently reviewed, PASS for design.
The guarded Go capture is implemented at `f6ea1353`. Initial independent review
found a raw-buffer lifetime blocker despite passing tests; an explicit KeepAlive
fix and re-verification are in progress. The offline analyzer is being verified
separately.
No enabled site capture, preflight, profile or matrix has run. Spectackle research
`R-01M23VWNB5FRP` is consumed by implementation tasks `T-01M24137G7EHM` and
`T-01M241R5P3EH8` and follows the
[complete fixed-count diagnostic](../m2-control-alloc-diagnostic-20260909/README.md).
Both original GELU candidate rejections remain unchanged.

`research.txt` preserves the initial source-backed proposal, not an approved
measurement protocol. It traces unchanged control/output/pool paths and pinned
Go 1.27.1 profiling semantics. A separate rate-1 experiment would be perturbed
diagnostic evidence, not interchangeable with unprofiled qualification.

`plan-review.txt` retains the fresh independent verdict: required corrections,
not PASS as originally written. `protocol.md` incorporates those corrections
without editing the initial research report. In particular:

- Public `runtime.MemProfileRecord` has counts and bytes but no ObjectSize.
  With `inuseZero=true`, this SDK's internal enumeration can return zero-active
  buckets. Size derivation needs a zero/unknown policy; 32-PC truncation also
  requires aggregation of visible-key collisions without discarding raw rows.
- Cross-binary matching by absolute source line can misidentify the same
  allocation after preceding source insertions. Raw locations must be retained,
  but a comparison key must avoid manufacturing differences from line shifts
  and clearly disclose any coarser function-stack aggregation.
- A noinline caller tag cannot appear in persistent pool-worker stacks. Keep
  caller, worker-temporally-associated, and all other process stacks separate.
- Immediate MemStats, GC-published profile deltas, and serialization tail are
  different accounting windows. Capture raw post records before any output
  serialization, and do not assert that profile sums equal immediate totals.

These are prospective instrumentation design concerns, not observed causes of
the completed Softplus differences. No profile, site attribution, gate revision,
noise subtraction, or runtime promotion is claimed here.

`corrected-review.txt` is the separate follow-up PASS. It leaves instrumentation,
fixed buffer capacity, schema, synthetic tests, and matched-build verification as
mandatory prerequisites before any run. The corrected design explicitly keeps
all prior rejections intact.

`schema.md` specifies the raw capture artifact format and fixed 65,536-record
buffers. `analysis-schema.md` specifies strict per-invocation integer validation,
both collision layers, all stack groups, contributor retention and derived JSON.
`execution-prerequisites.md` lists remaining preflight, decoder, matched-build,
runner/manifest, complete-matrix and independent analysis gates. Successful pure
tests are not a substitute for these future execution checks.

The capture's final implementer checks pass on the pinned Go 1.27.1 SDK: focused
default/SIMD tests, disabled diagnostic skips with no artifacts, full CPU short
suites, both vet modes, and focused race. The initial numeric-token test failure
and subsequent fixes are retained byte-identically in `capture-implementation.txt`
(SHA256 `c29dc9e5f251d58bbb5d1b788ba5be086570987294144fc264ee7b0d8b10a0fa`).
These passing tests do not override the independent lifetime finding: the raw
buffers must be kept live until all serialization finishes. No runtime kernel
change or speedup is part of this diagnostic implementation.
