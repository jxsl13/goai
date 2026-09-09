# M2 scalar F64 GELU backward direct-call experiment

**Rejected: the runtime specialization did not meet the predeclared gate.**
The production runtime is restored byte-for-byte; only tests, benchmark
harness, specifications, and evidence are retained.

This experiment is separate from the merged Go 1.27.1 compatibility rebuild.
It tests whether specializing the scalar GELU backward driver removes enough
per-element callback overhead to justify a production change. The exact
`geluGradF64` expression, its scalar function boundary, existing SIMD route,
materialization, allocation ownership, and scheduling policy must remain intact.
No approximation or incumbent-leadership claim is made here.
The generalizable function-value dispatch opportunity is tracked in
[perfscan issue #963](https://github.com/jxsl13/perfscan/issues/963), separately
from the generic interface/dictionary case in issue #905.

## Initial characterization

The unchanged runtime source is `137b335597640930158243753999aea0f6f9d9be`,
identical to merge commit `06993c9ef8c55bcfe132dca521654fde4f23a020`.
The host is an Apple M2 Pro, darwin/arm64, macOS 26.5.1 (25F80).
The official Go 1.27.1 toolchain used `GOTOOLCHAIN=local`, `CGO_ENABLED=0`,
`GOEXPERIMENT=simd`, and `GOARM64=v8.0`.
The frozen initial CPU test binary has SHA256
`fe3678cb26abf688919cb0f3ed8773bf3f05dce54bed8c70618a4de0a9fbcf09`.

`baseline.txt` retains nine samples per cell at 500 ms, after a discarded
one-iteration warmup. Each sample measured GOMAXPROCS 1 before 12.
This is baseline characterization, not an interleaved old/new comparison.

| Existing complete operation | GOMAXPROCS | Median ns/op | Median B/op | Median allocs/op |
| --- | ---: | ---: | ---: | ---: |
| GELU backward, 262144 elements | 1 | 3954529 | 2097480 | 5 |
| GELU forward, 256x2048 | 1 | 5267461 | 4194600 | 5 |
| GELU backward, 262144 elements | 12 | 800383 | 2097534 | 6 |
| GELU forward, 256x2048 | 12 | 961250 | 4194652 | 6 |

A separate five-second backward CPU profile at GOMAXPROCS 1 attributed 35.01%
of flat samples to `math.erf`, 24.74% to `math.archExp`, 11.32% to
`activationBackwardF64.func1`, and 4.61% to `geluGradF64`.
These shares motivate an attribution experiment; they do not predict its gain.
The profile was not collected concurrently with baseline measurements.
To reproduce, run the existing backward benchmark with `-test.benchtime=5s`
and `-test.cpuprofile=profile.pprof`, then use Go 1.27.1
`go run cmd/pprof -top profile.pprof`.

## Predeclared gate

Spectackle research `R-01M236QP4YEFD` is consumed by proposal
`P-01M237J19FFE5` and task `T-01M237QTE7E8E`.
The three `SCALAR-F64-GELU-DIRECT-*` contracts require exact scalar semantics
and three independent alternating count-seven campaigns:

- Target: 262144-element complete CPU backward at GOMAXPROCS 1 must improve
  by at least 1.05x, with p below 0.05, in every campaign.
- Controls: 2048 elements at GOMAXPROCS 1 and 12, and 262144 elements at 12,
  must have no reproducible time regression above 3% or allocation increase.
- Small effects must exceed within-arm noise; inconclusive measurements are
  not wins. Retain every campaign, including unfavorable ones.

`run.sh` operates on prebuilt old/new binaries containing the same
`BenchmarkGELUBackwardF64DirectDispatch` harness and seeded input construction.
It discards one-second warmups before each campaign, reverses the initial
arm and GOMAXPROCS order between campaigns, and alternates old/new per pair.
It records binary hashes and campaign/pair/arm/GOMAXPROCS markers.
No local builds, tests, profiles, or other benchmark campaigns may overlap.

The one-ULP mutation oracle must fail before any rewrite. The unchanged
scalar expression must remain exact against the reference for finite results
and signed zero, with matching nonfinite classes and input immutability.
Existing SIMD tolerances must not be widened. A failed performance gate
requires reverting the runtime candidate and preserving the rejection evidence.

## Frozen control and mutation

The test-only control commit is
`d0599a53ac51ded818526766c16321fcdcaa184e`. Before runtime edits,
`go test -c ./backend/cpu` with the exact Go 1.27.1 SIMD settings above produced
`gelu-direct-old.test`, SHA256
`d3c1e9c78ad70f170fcd39988856cc671dfd2fd63132f375bf1f1513f4213636`.
The shared harness file `backend/cpu/gelu_bwd_direct_test.go` has SHA256
`b2211d70284afa42eb2c5354e63f35b6f9e5f0f70fcd3765fb55ce7b6979b86c`.
The pre-change runtime file `backend/cpu/activation_bwd_f64.go` has SHA256
`ee4cc131844f4e4e259fd22c03d118a3a8d1a136d3f4135f88a645739852646b`.

Before the rewrite, temporarily replacing the scalar return with
`math.Nextafter(g*(phi+x*pdf), math.Inf(1))` failed both
`TestActivationBackwardF64CPUMatchesRef` and `TestCPUGeluBackwardCrossReference`
at their first GELU comparison; `mutation-one-ulp.txt` retains the failures.
The original runtime source was restored byte-for-byte before freezing the
control. Focused correctness passed in default and SIMD builds at GOMAXPROCS
1 and 12, including small/ragged/parallel shapes, special values, views,
input/output ownership, mixed-gradient fallback, validation errors, and recorder
counts. No numerical tolerance was widened.

## Candidate and result

Candidate source commit:
`56dc0b030e9c5587db1ea9a5d2a7a08d868105b0`.
The candidate binary, rebuilt from that commit, has SHA256
`2d71e0440d732089172f73e2d6b1b0ea887c58fbd21b52a153c4925491eca7ac`.
It is byte-identical to the independently tested preliminary candidate.
Its runtime file SHA256 is
`7ba7eb1a336ef24d5aff5ab83e80c4062230f735f3bf2d737cb5226ebd3b45ed`.
The shared harness SHA256 and all compiler settings match the frozen control.

`paired.txt` retains all 168 records, exactly seven per
campaign/arm/GOMAXPROCS/size cell. No builds, tests, or profiles ran concurrently.
`benchstat.txt` contains all timing, byte, and allocation comparisons, generated
with `benchstat -table campaign -col arm -ignore pair paired.txt` from
`golang.org/x/perf@v0.0.0-20260709024250-82a0b07e230d`.

The declared 262144-element, GOMAXPROCS 1 target failed in every campaign:

| Campaign | Old median ns/op | Candidate median ns/op | Old/new ratio | Time p |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4037885 | 3992276 | 1.0114x | 0.383 |
| 2 | 4032477 | 4012796 | 1.0049x | 0.710 |
| 3 | 4267130 | 4102917 | 1.0400x | 0.620 |

None reached 1.05x or p below 0.05. Within-arm target ranges divided by the
median were 15.32%/14.01%, 44.94%/9.08%, and 15.09%/43.67% (old/new),
also exceeding the near-5% noise expectation for a small claimed gain.
The parallel controls were especially noisy. Only the second campaign's small
GOMAXPROCS 12 control had a significant time difference (-2.76%, p=0.007);
that isolated cell does not satisfy the target gate.
Allocation counts remained unchanged. The single-worker cells saved 16 B/op,
which does not override the predeclared complete-operation latency requirement.

This rejects promotion under the measured conditions; it does not establish
that direct calls are universally neutral, slower, or equivalent. No samples
were dropped or campaigns replaced. No incumbent win is claimed.
The useful generalization for perfscan is to flag surviving indirect calls as
measurement obligations, not automatically beneficial rewrites, especially
when scalar transcendental work dominates.

An independent verifier recomputed all 168 records, all 84 successful benchmark
invocations, every cell median, paired direction, and two-sided Mann-Whitney
probability by enumerating the 3432 rank assignments. The complete sequence
matches `run.sh`. Target faster-pair counts were 5/7, 4/7, and 4/7.
One large-parallel pair allocated 27 more bytes in the candidate, despite lower
median bytes, so this record does not claim every allocation measurement improved.
The independent result also rejects promotion; it does not prove a slowdown
or a compiler defect.

Independent correctness verification passed the focused default/SIMD by
GOMAXPROCS 1/12 matrix, full CPU default/SIMD suites, full pure-Go build,
focused SIMD race tests, and Linux AMD64 SIMD test compilation.
Disassembly confirmed indirect `CALL (R1)` became a direct `geluGradF64` call;
the scalar function was not inlined, and its existing fused multiply-add
instruction was unchanged. Correctness alone did not justify promotion.
