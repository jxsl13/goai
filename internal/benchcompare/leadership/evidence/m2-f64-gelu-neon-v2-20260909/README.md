# M2 F64 GELU uniform-small follow-up — September 9, 2026

Status: independent correctness and code-generation verification passed.
Performance qualification is still pending; no V2 promotion decision yet.
The rejected [V1 experiment](../m2-f64-gelu-neon-20260909/README.md) and its frozen
binary remain immutable. This follow-up must pass the same gates; it does not
waive the small-input regression or claim external-library leadership.

- Parent: `P-01M23BZPZNF9V`; task: `T-01M23JPZT8FDM`.
- Independent plan reviewer: `m2-gelu-v2-plan-review` (PASS).
- Shortcut contract: `ARM64-F64-GELU-UNIFORM-SMALL-001`.
- Runtime/test commit: `361955ac35eaada696045f2cf828817b650174ec`.
- Build HEAD: `bd8faf5f69eb53bd3f0f696438b1671e8e625834` (spec-only follow-up).
- Frozen V2 binary SHA-256:
  `b768ca10228a7487aad1cbef3a658c99cb21078b5059b815fa690b0d9bcc7614`.
- V2 runtime source SHA-256:
  `72c79a0bf18257f19c38206ae6f206f8aa1227803d8e881fc65d068192075c35`.
- Formatted test source SHA-256:
  `cc87da8e6407080c7d931a9917b364adbdbff82698a631f30b16cb6e30abf7ca`.
- Original control source: `dd1e779eb085bb621ed5dafff0a4351636b6e656`.
- Control binary SHA-256:
  `c95691bc1a7289094bc523250fde7a9cae614756635b78701e9c306962b246cb`.
- Shared benchmark SHA-256:
  `b53599699510a31f3a2d086f08f1b11b968b9ad1f25941e75c8a1f62fc49c9f3`.
- Toolchain: Go 1.27.1, Darwin ARM64/v8.0, `GOEXPERIMENT=simd`,
  `GOTOOLCHAIN=local`, `CGO_ENABLED=0`, Apple M2 Pro.

## Hypothesis and scope

Return the already-computed small-region erf rational before exp/P/Q, only when
both lanes satisfy the existing strict `abs(y)<1` predicate. Mixed pairs and
exact boundaries retain V1's complete path. Coefficients, scalar twins, backward
exponential operation order, eligibility, fallback, output allocation, dispatcher,
and numerical bounds remain unchanged. Mask extraction and branching can cost
time or alter register allocation; performance is not assumed.

New explicit pair-permutation tests must pass before the runtime edit. A valid
compiled mutation inside the early return must fail an assertion and be restored
before committing, lifecycle writes, or final builds. Independent verification
reruns the full declared correctness/build/race/cross-build checks from the diff.

The [initial pair/body-tail tests](test-first-v1.txt) passed on unchanged V1.
Review added reversed both-small, signed-zero, and signed-subnormal pairs.
Two [compiled mutation checks](mutations.txt) then failed actual assertions:
corrupting the early-return result and allowing either lane to trigger it.
Both were restored before any committing operation resumed. A subsequent
import-order formatting repair is separately disclosed in the mutation record.
The production change is five new shortcut lines plus reuse of the same mask.
The parent checked the frozen binary metadata and all source/binary hashes.
The [emitted helper](erf-codegen.txt) extracts both mask lanes and branches before
exp/P/Q; the helper contains no calls or stack stores. This inspection establishes
the intended code path, not its performance. The fresh independent
[verification report](verification-report.txt) and [raw checks](verification.txt)
passed full default/SIMD CPU tests, focused process-count matrix, builds, vets,
AMD64 SIMD cross-compilation, focused race tests, formatting and diff checks.
Builds emitted disclosed module-stat-cache permission warnings but exited zero.
An initially incorrect helper-size count in the independent report was caught
and corrected against exact symbol boundaries: 880 to 896 bytes including
padding, a 16-byte increase. This is not a performance claim.

## Unchanged performance gates

Use the exact [V1 runner](../m2-f64-gelu-neon-20260909/run.sh), unchanged original
control, and a newly pinned V2 binary at a distinct path. Retain every sample in
three isolated alternating count-seven campaigns at `GOMAXPROCS=1` and `12`.
Run fixed controls separately, with no concurrent build/test/profile workload.

The [analysis script](analyze.rb) differs from V1 only in its required candidate
SHA-256. It retains exact invocation/order/cell/PASS checks and exact rank tests.
The unchanged V1 input should be rejected by this V2 analyzer, not silently
accepted as V2 evidence.

Every large public forward/backward active/mixed cell in each campaign must be
at least 1.25x serial and 1.05x parallel faster with `p<0.05`. A reproducible
small-input or fixed-control slowdown above 3%, or allocation increase, vetoes
promotion. Analyze each cell and campaign independently; neither a geomean nor
a selective rerun can substitute for the gate. Fresh independent performance
review and successful final CI jobs **and steps** are required before merge.

Generalizable opportunity: [perfscan #966](https://github.com/jxsl13/perfscan/issues/966).
The [completed V1 follow-up](https://github.com/jxsl13/perfscan/issues/966#issuecomment-5605855894)
establishes the regression, not causal attribution or a measured V2 improvement.

The disassembly also motivated [perfscan #967](https://github.com/jxsl13/perfscan/issues/967)
about repeated coefficient-address materialization. No coefficient-layout
candidate was implemented or measured; V2 results must not be attributed to it.

## CI setup observation

Run `34383399053` at `bd8faf5f` failed CUDA/Vulkan Ubuntu setup before compilation
because Google's Chrome package index failed its expected SHA-256 check on the
runner. The same failure recurred on the failed-job retry (attempt 2). The
[initial raw excerpts](ci-setup-failure.txt) are retained; no integrity check was
disabled and neither attempt counts as passing. Local qualification proceeds
independently; final CI success remains mandatory before merge.
