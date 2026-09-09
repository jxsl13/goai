# M2 autograd bounds-check experiment

Status: **unqualified experiment; no performance result yet**.

This follows the compiler-confirmed PS6093 application reported in
[perfscan issue 904](https://github.com/jxsl13/perfscan/issues/904#issuecomment-5609787036).
It does not repeat the older reference-backend bounds-check optimization.
The old and candidate arms will share benchmark source and compiler settings.
Removing a compiler check is a hypothesis, not proof of a useful speedup.

## Source and scope

The baseline is main commit
`c6afe9e4ba8b2a46c254953c921cf97ddf4f72c6`.
Its `autograd/vjp_elementwise.go` SHA-256 is
`07adbf223e44d862d5f9e3e38a743eff54def329280bb8b76aa0ffe3f9b62945`.
Proposal `P-01M246EFHMFR3` and task `T-01M246GYK2F9N` cover only the
F32/F64 typed loops in `unaryVJP`, `reluVJP`, `tanhVJP`, and `sigmoidVJP`.

The intended change gives the compiler equal-length slice proofs before the
hot loop. A historical cold path must preserve behavior when the proof cannot
be established. In particular, a short gradient with all-negative ReLU inputs
must not acquire an unconditional bounds panic. Callback order, floating-point
expressions, F64 intermediates in F32 gradients, and dtype fallbacks remain
unchanged. Empty tensors and prefix views with oversized backing storage must
remain safe and must not process storage outside the logical tensor extent.

Exp, GELU, and SiLU retain their backend dispatch. Selection, division, clip,
where, and row-wise softmax are outside this experiment. The selection and
division loops use backing-slice iteration rather than these helpers' Numel
bounds and need a separate view-semantics investigation before a bulk rewrite.
This is not a claim that every PS6093 finding has been addressed.

The earlier GELU branch is not a baseline. Its draft PR 1249 passed its final
CI run, but its frozen performance gate still failed, so that runtime is not
eligible for merging. This experiment starts independently from clean main.

## Correctness gate

Before the runtime change, probe the original suite with compiling, reached
one-ULP and index/condition mutations. Add missing historical-algorithm exact
oracles while production source is still unchanged, then demonstrate that the
oracles reject the mutations. A compilation error does not count as a detected
numerical defect. New tests must invoke the actual production helper.

Cover both floating-point dtypes, signed zero, subnormals, infinities, NaN
payloads, zero/scalar/tail/large extents, independently varied dense/prefix/
offset/transposed layouts, mixed dtypes and F16 fallback, and short-input
callback/panic behavior. Compare raw floating-point bits against the original
algorithm compiled on the same architecture; do not normalize NaNs or replace
the old algorithm with a mathematically equivalent formula.

The source review and focused/full short autograd tests run in default and
SIMD builds, with a separate focused race run. Optimized compiler BCE output
and ARM64 disassembly must distinguish entry checks and deliberate cold-loop
checks from the intended proof-backed hot loops. A fresh-context verifier
repeats the declared verification independently.

## Frozen measurement gate

Use the official Go 1.27.1 SDK, `GOTOOLCHAIN=local`, `CGO_ENABLED=0`, both
default and `GOEXPERIMENT=simd` builds, and GOMAXPROCS 1 and 12 on this Apple
M2 Pro. Prebuild old/candidate binaries from matching test source. Do not build,
test, profile, or run another owned benchmark concurrently with measurements.
Record source and binary hashes and the full environment before running.

- Direct registered VJP: ReLU, Tanh, Sigmoid, Log; F32/F64;
  Numel 0, 31, 2048, and 262144. Include result allocation, materialization,
  callback/dispatch work, and return overhead.
- Taped backward: the same operations and dtypes at Numel 2048 and 262144.
  Time the complete seeded `BackwardGrad` traversal and gradient-map replacement.
  Forward construction is outside this specifically labeled measurement.
- Whole training controls: the existing Adam F32/F64 and SGD F64 train-step
  benchmarks, including forward, loss, backward, and optimizer work.

Run three campaigns. Each campaign contains seven paired old/new invocations,
alternating order. Discard the first sample of each benchmark in every
invocation, retain the next sample, and keep all raw output including warmup.
This yields seven retained samples per arm and cell per campaign. Reverse
initial arm and GOMAXPROCS order between campaigns. Use identical measurement
durations in both arms; record those durations before the run.

GOMAXPROCS 1 target cells in each build and campaign must meet all of:

- Direct ReLU at Numel 2048 and 262144: median old/new ratio at least 1.05x.
- Taped ReLU at Numel 262144: median old/new ratio at least 1.03x.
- Each target difference has a two-sided p-value below 0.05.

All other cells, including GOMAXPROCS 12, are controls: no reproducible time
regression above 3% and no median B/op or allocs/op increase. For sub-10% claims,
within-arm spread must remain near 5%; noisy campaigns remain recorded but do
not qualify. No replacement of unfavorable samples, allocation-noise
subtraction, or post-result gate changes. A shorter diagnostic pilot can reject
the candidate early but cannot qualify it for shipping.

Retain every command, exit code, raw result, warmup exclusion, compiler
artifact, and benchstat comparison. A rejection removes the candidate runtime;
useful exact tests and the evidence can remain. This comparison can establish
an improvement over pinned GoAI, not leadership over an external library.

## Publication

Keep a dedicated draft PR while qualification is incomplete. A merge requires
the correctness and performance gates plus all exact-head CI jobs and executed
steps, including soft SIMD lanes. Only then delete the verified merged remote
feature branch. Report generalizable findings back to the existing perfscan
issue, preserving the distinction between detected opportunity and measured
end-to-end benefit.

## Runner and evidence format

`run.rb` freezes a 200 ms sample duration for both arms, before measurement.
Supply separate old/new binary directories containing
`default-autograd.test`, `simd-autograd.test`, `default-nn.test`, and
`simd-nn.test`. These must be built from the same test source with the pinned
Go 1.27.1 settings above. A pilot only needs the two default autograd binaries.
The output directory must not exist; an existing directory is never reused.

```sh
ruby run_test.rb
ruby run.rb /absolute/old-binaries /absolute/new-binaries /absolute/new-pilot pilot
ruby run.rb /absolute/old-binaries /absolute/new-binaries /absolute/new-campaign qualify
```

The pilot runs one seven-pair, GOMAXPROCS 1 default-build campaign over eight
ReLU direct/taped cells. It cannot qualify the candidate. Full mode runs all
48 autograd cells plus three training controls in each required build and
GOMAXPROCS setting over three campaigns: 336 invocations, with 8568 retained
rows and the same number of excluded first-sample rows.

Each invocation retains stdout, stderr, exact arguments, runtime environment,
timestamps, exit/signal status, and output hashes. The manifest pins binary and
runner hashes, planned order, sample duration, controlled runtime settings, and
completion/failure state. Every child uses only the recorded `CHILD_ENV` plus
its recorded GOMAXPROCS; all other inherited variables are unset. This avoids
both unrecorded runtime settings and accidental disclosure of user secrets.
The fixed environment sets GOGC=100, GOMEMLIMIT=off, GODEBUG empty, and
GOTRACEBACK=single, with C locale and `/usr/bin:/bin` PATH.
Binaries are rehashed before and after each invocation and at completion.
Post-run mutation or removal retains raw process evidence but fails the run.
`retained.txt` contains only validated second samples, with arm/build/scope/
campaign/pair/GOMAXPROCS labels. Missing, extra, or duplicate cells, mismatched
host/package headers, nonfinite metrics, and unsuccessful processes fail the
run without discarding earlier or partial raw evidence. Allocation metrics
are validated as decimal integers without a float round-trip.

The runner does not decide qualification or assert source comparability.
Inspect source/build manifests and compiler evidence separately, then compare
retained rows with the pinned benchstat and apply every frozen target/control
and noise requirement. Its tests use synthetic subprocesses, not performance
measurements. A completed pilot or runner test is not a speedup.

## Test-first baseline checkpoint

The runtime is still the original source hash above. The added
`autograd/vjp_bounds_internal_test.go` is pinned to SHA-256
`bd26d0195b02e23aa1406f01f7df25fff5aa65ea81fea0c3bae9f31d3359648c`.
The new oracle passes in default and SIMD builds. Default autograd short tests,
mdlint, and apicheck pass; the CGO-enabled default autograd short suite passes.
The full SIMD autograd short suite fails the pre-existing F64 sigmoid-focal
bit-parity and WKV exact-parity tests. A separate pristine worktree at baseline
`c6afe9e4ba8b2a46c254953c921cf97ddf4f72c6`, without the new oracle file,
reproduces both failures with the same values. They are unresolved baseline
failures, not a passing correctness gate or an excuse to skip SIMD validation.

`source-ready-report.txt` preserves the test author's mutation history,
including explicitly abbreviated early output. The separate `reach-*`,
`finite-one-ulp-oracle.log`, and `original-source-oracle-final.log` files retain
the corrected raw probes. Compile failures are not counted as caught mutants.
The finite-normal tanh fixture detects a one-ULP perturbation; ReLU zero-branch
mutations are rejected for both dtypes. Valid fixtures require successful
outputs, so matching failures cannot masquerade as valid numerical parity.

The runner's first independent review failed on final-invocation binary
mutation, contradictory extra headers, and incomplete inherited environment
capture. The initial verdict and supplemental output are preserved as
`runner-review-initial-fail.txt` and `runner-review-initial-supplement.txt`.
Regression fixes are under independent re-review; no live timings have run.

Benchmark `MB/s` is based on nominal output bytes from `SetBytes`, not measured
memory bandwidth. Qualification uses ns/op, B/op, and allocs/op.
