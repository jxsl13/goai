# M2 F64 GELU intrinsic experiment — September 9, 2026

Status: V1 rejected for production promotion. All large public targets improved,
but a repeatable small-input regression violates the declared control gate.
The draft branch is continuing qualification; V1 must not be merged as-is.

## Scope and pins

- Base: `b9c059464edb4a339073c75ea8037c5472bf9aea`.
- Host: Apple M2 Pro, Darwin ARM64.
- Compiler: Go 1.27.1, `GOTOOLCHAIN=local`, `CGO_ENABLED=0`.
- Measurement build: `GOEXPERIMENT=simd`.
- Research: `R-01M23B9YZ1E16`.
- Proposal: `P-01M23BZPZNF9V`; task: `T-01M23C5EWPE8P`.
- Design and numerical boundaries:
  [ADR-0031](../../../../../docs/decisions/ADR-0031-arm64-f64-gelu.md).

The shared `BenchmarkVGELUF64NeonBoundary` harness distinguishes preallocated
leaf execution from public `backend.Execute`, including output allocation. It
tests forward and backward GELU, active-range and mixed-erf-region distributions,
and 2,048 and 262,144 elements. The candidate must retain the exact same harness
bytes as the frozen control.

## Reproduction protocol

Build control and candidate test binaries from their recorded source commits
with the same compiler and build environment. Do not run builds, tests, profiles,
or other CPU-heavy workloads concurrently with measurement. Run:

```sh
bash run.sh /absolute/path/to/control.test /absolute/path/to/candidate.test
bash run.sh /absolute/path/to/control.test /absolute/path/to/candidate.test controls
```

The runner prints binary SHA-256 hashes and campaign, pair, process-count, and
arm labels. It excludes one-second warm-ups per cell and collects three
order-balanced, count-seven campaigns at `GOMAXPROCS=1` and `12`. All measured
records must be retained, including unsuccessful or noisy campaigns.

Analyze each campaign independently, preserving the benchmark's boundary,
direction, distribution, size, and process-count dimensions. The declared large
public-operation gate requires at least 1.25x serial and 1.05x parallel speedup,
each with `p<0.05`, for both directions and both distributions in every campaign.
Control regressions and allocation increases can veto promotion.
The fixed non-target controls are F64 Sigmoid (65536 elements), Softplus
(262144), and SiLU backward (262144), using existing unchanged public-operation
harnesses and the same three-campaign protocol. They run separately, never
concurrently with the GELU measurements.

This internal comparison is not an external-library leadership claim. The
previous scalar callback experiment was rejected; its profile is only motivation
for investigating the transcendental cost here.

The short two-pair, 200 ms [pilot](pilot.txt) uses [pilot.sh](pilot.sh), includes
only the four large public targets, and observed faster candidate samples in
both arm orders. It is diagnostic, not statistical promotion evidence. macOS
background services were consuming multiple cores before it ran; no Go builds,
tests, profiles, or other agent benchmark runs overlapped it. The full campaigns
and non-target controls below supersede this diagnostic, with all samples retained.

## Complete V1 measurement and decision

All 84 GELU invocations completed: exactly 1344 records, seven per
campaign/arm/process-count/benchmark cell. The separately run non-target control
set completed another 84 invocations and 252 records. No samples or campaigns
were dropped. Binary and shared-harness hashes remained unchanged.

Reproduce analysis with:

```sh
ruby analyze.rb paired.txt controls.txt > analysis.csv 2> analysis-audit.txt
benchstat -table campaign -col arm -ignore pair paired.txt controls.txt > benchstat.txt
```

The audit checks binary hashes, complete invocation order, PASS markers, exact
cell membership, and seven pairs. Its exact two-sided permutation rank test
enumerates all 3432 assignments and uses average tied ranks. Deliberately
removing the final PASS or changing the candidate hash is rejected; retained
failure logs are `analyzer-missing-pass.txt` and `analyzer-wrong-hash.txt`.
The CSV reports time, bytes, and allocation counts separately. Descriptive flag
counts do not automatically waive one- or two-campaign control findings.

All 24 large public target/campaign cells met their speedup and significance
requirements. Ranges below cover the three campaigns, not confidence intervals:

| 262144-element public operation | Serial speedup | Parallel speedup |
| --- | ---: | ---: |
| Forward, active | 1.335–1.340x | 1.258–1.269x |
| Forward, mixed | 2.785–2.795x | 2.216–2.253x |
| Backward, active | 1.378–1.383x | 1.394–1.413x |
| Backward, mixed | 2.588–2.591x | 2.196–2.240x |

Every large target has exact nominal `p=0.0005827506` (benchstat displays 0.001).
Nevertheless, small active-range forward Execute regressed in every campaign:

| Campaign | GOMAXPROCS=1 slowdown | GOMAXPROCS=12 slowdown |
| --- | ---: | ---: |
| 1 | +68.95% | +74.31% |
| 2 | +69.98% | +75.96% |
| 3 | +72.75% | +76.65% |

These six comparisons have the same nominal `p=0.0005827506`; the corresponding
small direct-leaf controls also regressed in all three campaigns. This is a
decisive veto under the 3% control rule. Neither a favorable geomean nor the
large-input wins can override it.

The fixed non-target controls had no repeatable significant time regression
above 3%. Two isolated parallel medians increased by 2 B/op: campaign two's
SiLU backward (4194701 to 4194703; exact nominal `p=0.51457`) and campaign
three's Sigmoid (524613 to 524615; `p=0.01457`). Neither repeated in the other
campaigns; allocation-count medians were unchanged. Both findings are retained,
not silently treated as a universal allocation improvement.

macOS background services were active before measurement and some parallel
samples were noisy. These results describe the recorded host conditions, not
universal latency or external-library/model leadership. The fresh independent
[performance recomputation](verifier-performance.txt) validated the complete
protocol and rejected promotion under the control veto. It found the small-input
regression reproducible and statistically decisive, not an isolated ambient
load excursion. Candidate CI at `17a4ff45` passed all 16 jobs and every individual
step, including soft SIMD lanes; see [ci-candidate.json](ci-candidate.json).

## Bounded follow-up

The current helper eagerly computes the exp and complementary rational before
selecting the small region. A prospective uniform-small fast path can return
the already-computed small rational when **both** lanes satisfy the existing
strict `abs(y)<1` predicate. The pinned ARM64 API provides `Mask64x2.ToInt64x2`
and constant-index `Int64x2.GetElem(0/1)`; it has no `Mask.All` method.
This idea is tracked as [perfscan #966](https://github.com/jxsl13/perfscan/issues/966).
It is implemented in the separately qualified
[V2 follow-up](../m2-f64-gelu-neon-v2-20260909/README.md), with no V2 performance
decision yet. The same numerical boundaries, scalar twins, frozen control/harness,
and full gates remain required; extra branches or altered register allocation
can still regress mixed inputs. V1's raw measurements and binary are unchanged.

## Verification record

The test-only control was frozen before runtime changes:

- Source commit: `dd1e779eb085bb621ed5dafff0a4351636b6e656`.
- Control binary SHA-256:
  `c95691bc1a7289094bc523250fde7a9cae614756635b78701e9c306962b246cb`.
- Shared harness SHA-256:
  `b53599699510a31f3a2d086f08f1b11b968b9ad1f25941e75c8a1f62fc49c9f3`.
- Binary metadata: `go1.27.1-X:simd`, Darwin ARM64/v8.0, `CGO_ENABLED=0`.

The parent independently checked those hashes, the unchanged production diff,
and the frozen focused numerical, view, fallback, and existing exact tests.
Those tests passed. The control's dense special-value fixture is an exact
whole-wrapper fallback oracle, not proof that a future vector path runs.

The default exact oracles reject a one-ULP mutation in both build modes;
the public routing tests reject disabling the dedicated SIMD gate in both
directions at 3, 4, 200003, and 262144 elements. Raw attempts, including an
invalid non-compiling NaN mutation that does not count as evidence and its later
compilable assertion-failing replacement, are retained
in [mutations.txt](mutations.txt). Candidate acceptance also
requires independent numerical, special-value, input-immutability, aliasing,
body/tail, allocation, routing, race, cross-build, and generated-code checks.
The retained measurements above reject production promotion of V1 despite its
passed correctness checks.

The fresh verifier passed full CPU tests, focused default/SIMD tests at process
counts 1 and 12, full-tree builds, CPU vet, AMD64 SIMD cross-compilation,
focused race tests, formatting, and diff checks. See
[verification.txt](verification.txt). The restored NaN audit passed its focused
source test, left production source and the frozen binary unchanged, and was
independently checked by the parent and verifier. Eligible numerical testing
observed maximum absolute error `4.440892098500626e-16` and maximum normalized
error `3.318506137718579e-16`; sampled errors are not a universal error proof.

Candidate runtime commit: `e766a961505adf071cae8642b8722082ebcaf48b`.
The test-only verifier-gap repair is `dc954bc9343fb56c1cb0d659ce98e5af43dddf5c`;
the frozen candidate built from it has SHA-256
`045826effd7c6aab639a1b4c9e0e162e4eca03b6aaef56761291c248492301d7`.
Its shared benchmark harness matches the control byte-for-byte. Initial
independent review required stronger whole-wrapper fallback and public SIMD
reachability coverage; those tests were repaired before measurement.

Generated code contains real two-lane arithmetic, with the exp polynomial
inlined into the erf helper. The wrappers still call that helper per pair and
spill/reload vector inputs across it. This is a performance risk to measure,
not proof of a speedup or compiler defect.

During an earlier test mutation, a Spectackle auto-commit captured the temporary
scalar change. It was pushed only to the draft branch, caught by CI, and restored
in `fc60945b7d8ab817e080aef3753da75d4d647e7d`; it never reached main.
[PR incident record](https://github.com/jxsl13/goai/pull/1249#issuecomment-5604611720).
The corrected checkpoint passed all 16 CI jobs and their individual steps.
New process contracts serialize mutations with every committing operation and
require immediate verification of the committed runtime diff before pushing.

## Further reading

- [Previous rejected scalar experiment](../m2-f64-gelu-direct-20260909/README.md).
- [Shared-transcendental findings](https://github.com/jxsl13/perfscan/issues/917).
- [Constant SIMD shift lowering](https://github.com/jxsl13/perfscan/issues/965).
- [Eager piecewise SIMD branches](https://github.com/jxsl13/perfscan/issues/966).
