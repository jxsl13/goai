# M2 F64 GELU intrinsic experiment — September 9, 2026

Status: control frozen; candidate implementation in progress. No candidate speedup or production
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

This internal comparison is not an external-library leadership claim. The
previous scalar callback experiment was rejected; its profile is only motivation
for investigating the transcendental cost here.

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

The implementer reported that the default exact oracles reject a one-ULP
mutation in both build modes; raw mutation records are being captured before
the runtime rewrite. Candidate acceptance also
requires independent numerical, special-value, input-immutability, aliasing,
body/tail, allocation, routing, race, cross-build, and generated-code checks.
Final commits, hashes, test outcomes, raw measurements, and the acceptance or
rejection decision will be added when those checks have completed.

## Further reading

- [Previous rejected scalar experiment](../m2-f64-gelu-direct-20260909/README.md).
- [Shared-transcendental findings](https://github.com/jxsl13/perfscan/issues/917).
