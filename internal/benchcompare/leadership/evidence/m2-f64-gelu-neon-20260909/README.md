# M2 F64 GELU intrinsic experiment — September 9, 2026

Status: control and candidate frozen; correctness verified, full measurement pending. No validated speedup or production
promotion is claimed by this record.

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
and non-target controls remain required, with all samples retained.

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
Final commits, hashes, test outcomes, raw measurements, and the acceptance or
rejection decision will be added when those checks have completed.

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
