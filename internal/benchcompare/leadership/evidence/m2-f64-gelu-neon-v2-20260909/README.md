# M2 F64 GELU uniform-small follow-up — September 9, 2026

Status: minimal candidate implemented; independent qualification pending. No V2
measurement yet.
The rejected [V1 experiment](../m2-f64-gelu-neon-20260909/README.md) and its frozen
binary remain immutable. This follow-up must pass the same gates; it does not
waive the small-input regression or claim external-library leadership.

- Parent: `P-01M23BZPZNF9V`; task: `T-01M23JPZT8FDM`.
- Independent plan reviewer: `m2-gelu-v2-plan-review` (PASS).
- Shortcut contract: `ARM64-F64-GELU-UNIFORM-SMALL-001`.
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
The production change is five new shortcut lines plus reuse of the same mask;
full correctness and generated-code verification must still complete.

## Unchanged performance gates

Use the exact [V1 runner](../m2-f64-gelu-neon-20260909/run.sh), unchanged original
control, and a newly pinned V2 binary at a distinct path. Retain every sample in
three isolated alternating count-seven campaigns at `GOMAXPROCS=1` and `12`.
Run fixed controls separately, with no concurrent build/test/profile workload.

Every large public forward/backward active/mixed cell in each campaign must be
at least 1.25x serial and 1.05x parallel faster with `p<0.05`. A reproducible
small-input or fixed-control slowdown above 3%, or allocation increase, vetoes
promotion. Analyze each cell and campaign independently; neither a geomean nor
a selective rerun can substitute for the gate. Fresh independent performance
review and successful final CI jobs **and steps** are required before merge.

Generalizable opportunity: [perfscan #966](https://github.com/jxsl13/perfscan/issues/966).
The [completed V1 follow-up](https://github.com/jxsl13/perfscan/issues/966#issuecomment-5605855894)
establishes the regression, not causal attribution or a measured V2 improvement.
