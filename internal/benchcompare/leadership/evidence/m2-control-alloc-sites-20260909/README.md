# Prospective control allocation-site investigation

Status: corrected prospective protocol independently reviewed, PASS for design.
The guarded Go capture and per-invocation Ruby analyzer have each passed separate
independent implementation verification. Initial reviews found a raw-buffer
lifetime blocker and an overflowed-JSON-exponent parser blocker; both original
FAIL reports are retained alongside the narrow fixes and follow-up PASS reports.
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
The initial tests did not override the independent lifetime finding. Commit
`3eb438b28dd0ac68a469848d5b3d72ab7ca305ea` adds explicit KeepAlive calls for both
raw buffers after final artifact serialization and close attempts. The complete
pinned verification was independently rerun and passed. Source SHA256 is
`b6583b360128ba53ef87f71bff0b705dbdb549e1de78a5115774aeab14017352`.

The analyzer initially passed 19 tests/272 assertions, but independent review
showed that Ruby 2.6.10 parses `1e999` as Infinity despite `allow_nan: false`.
Commit `e57d49aafa700e6428a1fcb6f2fad7b95a95e1ae` adds recursive finite-value
validation without weakening exact-integer schema fields. Follow-up independent
verification passed 20 tests/290 assertions and the original independent suite
(7 tests/77 assertions). `independent_verify_test.rb` preserves that supplemental
suite byte-identically; run it with Ruby from this directory.

Preserved review trail:

| Artifact | Verdict / purpose |
| --- | --- |
| `capture-verification-fail.txt` | Original independent lifetime FAIL |
| `capture-keepalive-fix.txt` | Narrow fix and full implementer rerun |
| `capture-verification-pass.txt` | Independent full follow-up PASS |
| `analysis-implementation.txt` | Initial implementation, failures and corrections |
| `analysis-verification-fail.txt` | Original independent nonfinite-parser FAIL |
| `analysis-nonfinite-fix.txt` | Narrow fix and full implementer rerun |
| `analysis-verification-pass.txt` | Independent full follow-up PASS |

The capture follow-up transcript contains literal Git diff context: leading
space-before-tab and a space-only context line are preserved, not source defects.
Its evidence-only diff check used `core.whitespace=-space-before-tab,-blank-at-eol`;
all other staged files passed the ordinary check. No repository configuration or
source whitespace policy was weakened.

These are implementation-only results. No runtime kernel change, measured speedup,
live profile, allocation-site attribution or GELU qualification is claimed.
