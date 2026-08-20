# M2 arm64 exact Abs and EAGLE Smooth-L1 fusion — 2026-08-20

## Claim boundary

This evidence establishes a measured GoAI winner matrix against its exact pre-tranche implementation and direct synchronous Metal route on one Apple M2 Pro. It does not claim universal cross-library leadership. Every comparison keeps dtype, shape, result semantics, output allocation, warmup count, and operation boundary fixed.

The shipped changes are:

- an exact arm64 F32 Abs kernel that replaces the F32-to-F64 scalar loop;
- an Abs-specific serial/parallel crossover at 262,144 elements;
- an M2 host-route ceiling of 16,777,216 F32 Abs elements in default and `GOEXPERIMENT=simd` builds;
- an unscaled `OpSmoothL1Core` plus exact fused VJP for CPU/reference backends;
- capability-gated EAGLE fusion that retains the composite graph on unsupported backends.

## Pinned environment

| Item | Value |
| --- | --- |
| Hardware | Apple M2 Pro, darwin/arm64, benchmark runtime reports 12 logical CPUs |
| OS | macOS 26.5.1 (25F80) |
| Go | go1.26.6 darwin/arm64 |
| Spectackle | v0.9.3; temporarily rebuilt with Go 1.26.6 for typed indexing |
| Base | `8bc05528c1212fc316e19109656cd818f0166cea` |
| Exact Abs checkpoint | `dde59882` |
| Fusion and route checkpoint | `9f93b5b9` |
| Contract checkpoint | `ac185194` |
| Protocol | 20 untimed warmups, 100 iterations, count=7, median of each seven-sample cell, three independent campaigns |

The Homebrew Spectackle 0.9.3 binary was built with Go 1.25.0 and disabled typed indexing for this Go 1.26 repository. Rebuilding the same release module with Go 1.26.6 restored 6,714 typed call edges; packaging report: [spectackle#274](https://github.com/jxsl13/spectackle/issues/274).

## Exact Abs semantic contract

The incumbent expression was `float32(math.Abs(float64(x)))`. Raw-bit probes establish this F32 contract:

- clear the sign of finite values, zero, infinities, and NaNs;
- preserve every NaN payload bit;
- set the F32 quiet bit on every NaN.

Examples include `0x7f800001 -> 0x7fc00001`, `0xff800001 -> 0x7fc00001`, and `0x7fa00001 -> 0x7fe00001`.

Two mutations were rejected before promotion. Vector `FABS` preserved signaling NaNs unchanged, while an initial mask/select rewrite retained only the quiet bit and destroyed the payload. The shipped 16-lane kernel clears the sign, classifies NaNs with unsigned magnitude comparisons against +Inf, and conditionally ORs the quiet bit. All raw-bit and all-length tests pass.

General analyzer finding: [perfscan#778](https://github.com/jxsl13/perfscan/issues/778).

## Complete CPU Abs operation

Control is the exact pre-tranche scalar implementation. Candidate includes production dispatch, output allocation, the exact NEON leaf, and the operation-specific crossover.

| Elements | C1 control ns | C1 candidate ns | C1 x | C2 control ns | C2 candidate ns | C2 x | C3 control ns | C3 candidate ns | C3 x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 3,157 | 1,325 | 2.383 | 3,169 | 1,264 | 2.507 | 3,464 | 1,328 | 2.608 |
| 65,536 | 44,511 | 26,500 | 1.680 | 43,793 | 27,154 | 1.613 | 45,543 | 26,678 | 1.707 |
| 131,072 | 77,248 | 45,628 | 1.693 | 78,705 | 45,518 | 1.729 | 80,498 | 45,884 | 1.754 |
| 349,440 | 126,474 | 89,460 | 1.414 | 130,393 | 85,708 | 1.521 | 130,652 | 85,370 | 1.530 |
| 524,288 | 154,185 | 102,795 | 1.500 | 153,106 | 102,636 | 1.492 | 156,003 | 101,388 | 1.539 |
| 2,097,152 | 346,265 | 248,663 | 1.393 | 330,535 | 246,528 | 1.341 | 346,979 | 247,531 | 1.402 |
| 4,194,304 | 562,830 | 442,285 | 1.273 | 570,916 | 443,672 | 1.287 | 564,406 | 424,511 | 1.330 |
| 8,388,608 | 1,099,269 | 921,601 | 1.193 | 1,103,670 | 918,790 | 1.201 | 1,093,688 | 930,022 | 1.176 |

All 24 paired cells exceed the 1.10 gate. Allocations are unchanged or lower; the serial-vector policy avoids the worker closure below 262,144 elements.

## M2 route extension before selector promotion

This benchmark bypasses the production ceiling and compares the exact CPU candidate directly with synchronous Metal. Values are speedup over direct Metal.

| Elements | Default C1 | Default C2 | Default C3 | SIMD C1 | SIMD C2 | SIMD C3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,194,304 | 3.853 | 3.752 | 3.754 | 3.862 | 3.697 | 3.895 |
| 8,388,608 | 2.815 | 2.800 | 2.847 | 3.320 | 2.846 | 2.900 |
| 16,777,216 | 2.983 | 2.999 | 3.008 | 2.989 | 3.016 | 3.108 |

All 18 cells exceed 1.10, justifying the separate 16,777,216-element Abs ceiling.

## Production selector after promotion

Control forces direct Metal; candidate uses the production selector. Values are control median divided by candidate median.

| Elements | Default C1 | Default C2 | Default C3 | SIMD C1 | SIMD C2 | SIMD C3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 215.662 | 250.149 | 220.404 | 169.430 | 202.506 | 187.049 |
| 65,536 | 7.732 | 6.710 | 7.552 | 7.874 | 7.706 | 7.921 |
| 131,072 | 4.005 | 4.113 | 4.350 | 4.604 | 3.927 | 3.955 |
| 349,440 | 3.142 | 3.439 | 3.216 | 3.096 | 3.157 | 3.185 |
| 524,288 | 3.198 | 3.686 | 3.200 | 3.290 | 3.212 | 3.285 |
| 2,097,152 | 3.141 | 3.406 | 3.211 | 3.195 | 3.103 | 3.237 |
| 4,194,304 | 3.895 | 3.843 | 3.845 | 3.921 | 3.840 | 3.722 |
| 8,388,608 | 2.868 | 2.836 | 2.838 | 2.816 | 2.818 | 2.834 |
| 16,777,216 | 3.022 | 3.010 | 2.844 | 3.002 | 3.008 | 3.029 |

All 54 production cells exceed 1.10. The weakest promoted large-shape cell is 2.816x at 8M and 2.844x at 16M.

## EAGLE Amdahl gate and fused Smooth-L1 core

An order-alternating same-binary control showed that the Abs leaf alone improved the complete EAGLE feature regression by only about 1.03x at 349,440 elements and 1.02x at 2,097,152, failing the standing 1.03 leverage gate robustly. The gate was not weakened; the six-pass elementwise chain was fused instead.

`OpSmoothL1Core` intentionally returns the unscaled identity `d*d - ReLU(abs(d)-1)^2`. EAGLE retains the existing mean and 0.5 scale outside the fused boundary. The active backend must provide the core kernel; otherwise EAGLE retains the composite graph and never forces an implicit CPU fallback.

The first closed-form fused derivative was rejected because it differed by one ULP in both F32 and F64. The shipped VJP reproduces the composite tape's multiply, fan-out, select, and accumulation order. Forward and backward match raw bits for signed zero, finite extremes, infinities, NaNs, and 300,000-element parallel tensors. General analyzer finding: [perfscan#779](https://github.com/jxsl13/perfscan/issues/779).

### Forward

| Elements | C1 control ns | C1 candidate ns | C1 x | C2 control ns | C2 candidate ns | C2 x | C3 control ns | C3 candidate ns | C3 x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 349,440 | 1,270,829 | 674,122 | 1.885 | 1,309,076 | 695,723 | 1.882 | 1,304,835 | 678,952 | 1.922 |
| 2,097,152 | 5,780,687 | 3,960,678 | 1.460 | 5,711,606 | 4,042,831 | 1.413 | 5,790,014 | 4,077,210 | 1.420 |

### Forward plus backward

| Elements | C1 control ns | C1 candidate ns | C1 x | C2 control ns | C2 candidate ns | C2 x | C3 control ns | C3 candidate ns | C3 x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 349,440 | 5,443,820 | 1,618,190 | 3.364 | 6,034,915 | 1,737,855 | 3.473 | 5,394,523 | 1,498,402 | 3.600 |
| 2,097,152 | 31,308,314 | 9,805,390 | 3.193 | 29,044,028 | 9,630,637 | 3.016 | 28,378,779 | 8,491,475 | 3.342 |

All 12 fused workload cells exceed 1.03.

## Reproduction commands

```sh
go test ./backend/cpu -run '^$' -bench '^BenchmarkAbsF32CPU/' -benchtime=100x -count=7 -benchmem
go test ./backend/metal -run '^$' -bench '^BenchmarkMetalAbsRouteExtension/' -benchtime=100x -count=7 -benchmem
GOEXPERIMENT=simd go test ./backend/metal -run '^$' -bench '^BenchmarkMetalAbsRouteExtension/' -benchtime=100x -count=7 -benchmem
go test ./backend/metal -run '^$' -bench '^BenchmarkMetalUnaryRouteCandidates/abs/' -benchtime=100x -count=7 -benchmem
GOEXPERIMENT=simd go test ./backend/metal -run '^$' -bench '^BenchmarkMetalUnaryRouteCandidates/abs/' -benchtime=100x -count=7 -benchmem
go test -tags goai_bench_control ./nlp -run '^$' -bench '^BenchmarkEagleSmoothL1AbsRoute/' -benchtime=100x -count=7 -benchmem
go test -tags goai_bench_control ./nlp -run '^$' -bench '^BenchmarkEagleSmoothL1TrainingStep/' -benchtime=100x -count=7 -benchmem
```

Metal commands require unsandboxed GPU access. The control backend is build-tagged out of production binaries.

## Validation

- exact forward parity: CPU and reference versus the incumbent composite;
- exact VJP parity: fused versus taped composite for special and 300K parallel tensors;
- numerical gradient check;
- race detector over CPU/autograd Smooth-L1 tests;
- default and SIMD Metal selector/parity tests;
- repository `make preflight`: pass;
- focused in-tree perfscan over every changed production file: no candidates;
- `git diff --check`;
- Spectackle typed reindex: 17,530 nodes, 32,590 edges, 6,714 typed calls, zero skipped;
- Spectackle drift audit: zero tranche drift records; repository-wide historical lint warnings remain.

The unshortened `go test ./...` passed every package except the deterministic pre-existing `TestDiffusionLMGrammarE2E`: CE `2.924 -> 0.085`, generation `"a sra s ee s a s"`, invalid grammar window. The output was reproduced byte-for-byte on this branch and on a detached exact `origin/main` worktree at `8bc05528`, so it is not attributed to this tranche.
