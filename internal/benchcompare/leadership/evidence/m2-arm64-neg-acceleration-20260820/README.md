# M2 arm64 exact F32 Neg acceleration — 2026-08-20

## Claim boundary

This evidence establishes a measured GoAI winner matrix against the exact pre-tranche CPU implementation and direct synchronous Metal route on one Apple M2 Pro. It does not claim universal cross-library leadership. Dtype, shapes, raw-bit semantics, allocation boundary, warmups, transfers, and synchronization are fixed between each paired control and candidate.

The retained changes are an exact arm64 F32 Neg kernel, a measured serial/parallel crossover at 1,048,576 elements, and a Neg-specific M2 host-route ceiling of 16,777,216 elements in default and `GOEXPERIMENT=simd` builds.

## Pinned environment

| Item | Value |
| --- | --- |
| Hardware | Apple M2 Pro, darwin/arm64 |
| OS | macOS 26.5.1 (25F80) |
| Go | go1.26.6 darwin/arm64 |
| Spectackle | v0.9.3, rebuilt with Go 1.26.6 for typed indexing |
| Base | `85aba5abfcf4a9b10f72d07895577b1723ce1334` |
| Implementation checkpoint | `e24f685e3f3232fd7e5e1f763ffde18ddf8b0a2d` |
| CPU/route protocol | 20 untimed warmups, 100 iterations, count=7, per-cell median, 3 independent campaigns |
| Workload protocol | 10 untimed warmups, 20 iterations, count=7, per-cell median, 3 independent campaigns |

Spectackle proposal `P-01M0GFFBN1F1N`, task `T-01M0GFJMPQE4G`, and rules `ARM64-EXACT-NEG-001`, `ARM64-EXACT-NEG-PERF-001`, `MEASURED-METAL-UNARY-ROUTE-001`, and `MEASURED-METAL-SIMD-UNARY-ROUTE-001` govern the change.

## Exact semantic contract

For every F32 lane, `outputBits = inputBits XOR 0x80000000`. The integer-domain implementation changes only the sign bit, preserving finite magnitudes, both zero encodings, infinities, signaling and quiet NaNs, and every NaN payload bit. The arm64 leaf processes 16 lanes per iteration with four `VEOR` instructions; the portable fallback implements the same raw-bit contract.

Tests cover lengths 0 through 257, unaligned subslices, destination guard regions, randomized raw bits, subnormals, both zero signs, infinities, and signaling/quiet NaNs. A separate test crosses the production parallel threshold. `go tool objdump` confirms the emitted four-vector `VEOR` loop.

General analyzer finding: [perfscan #780](https://github.com/jxsl13/perfscan/issues/780).

## Complete CPU Neg operation

Control is the exact pre-tranche scalar implementation. Candidate includes output allocation, production dispatch, the exact NEON leaf, and the operation-specific crossover. Speedup is control median divided by candidate median.

| Elements | C1 speedup | C2 speedup | C3 speedup |
| ---: | ---: | ---: | ---: |
| 2,048 | 2.142 | 2.159 | 2.213 |
| 65,536 | 1.897 | 1.859 | 1.760 |
| 131,072 | 1.811 | 1.668 | 1.597 |
| 349,440 | 1.768 | 2.114 | 1.959 |
| 524,288 | 1.572 | 1.552 | 1.525 |
| 2,097,152 | 1.475 | 1.528 | 1.632 |
| 4,194,304 | 1.303 | 1.277 | 1.378 |
| 8,388,608 | 1.221 | 1.243 | 1.261 |

All 24 paired cells exceed the 1.10 promotion gate; the weakest is 1.221x. Control and candidate execute in the same binary with identical inputs and operation boundaries.

## M2 route extension before selector promotion

This benchmark bypasses the production ceiling and compares the exact CPU candidate directly with synchronous Metal. Values are speedup over direct Metal.

| Elements | Default C1 | Default C2 | Default C3 | SIMD C1 | SIMD C2 | SIMD C3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,194,304 | 3.305 | 3.905 | 3.985 | 3.764 | 3.426 | 3.693 |
| 8,388,608 | 3.169 | 3.200 | 3.159 | 3.009 | 3.037 | 3.150 |
| 16,777,216 | 3.226 | 3.216 | 3.139 | 3.165 | 3.060 | 3.232 |

All 18 cells exceed 3.0x, justifying the independent 16,777,216-element Neg ceiling.

## Production selector after promotion

Control forces direct Metal; candidate uses the production selector. Values are control median divided by candidate median.

| Elements | Default C1 | Default C2 | Default C3 | SIMD C1 | SIMD C2 | SIMD C3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 213.614 | 241.261 | 214.202 | 240.101 | 226.730 | 177.787 |
| 65,536 | 7.774 | 7.953 | 8.409 | 7.603 | 7.312 | 7.254 |
| 131,072 | 3.905 | 3.944 | 4.344 | 4.166 | 3.797 | 4.068 |
| 349,440 | 3.879 | 3.728 | 3.584 | 3.725 | 4.323 | 3.872 |
| 524,288 | 3.153 | 3.172 | 3.119 | 3.106 | 3.254 | 3.573 |
| 2,097,152 | 3.098 | 3.079 | 3.152 | 3.114 | 3.234 | 3.360 |
| 4,194,304 | 3.742 | 3.980 | 3.747 | 3.901 | 3.908 | 3.625 |
| 8,388,608 | 2.827 | 2.822 | 2.856 | 3.083 | 2.816 | 2.835 |
| 16,777,216 | 3.012 | 3.024 | 3.012 | 3.010 | 3.015 | 2.834 |

All 54 production cells exceed 1.10. The weakest is 2.816x, so the new ceiling retains substantial margin in both builds.

## Workload Amdahl boundary

`SigmoidFocalLoss` was measured with only `OpNeg` switched between the frozen scalar control and production CPU kernel. Every other operation and the final raw-bit result remained identical.

| Elements | C1 x | C2 x | C3 x |
| ---: | ---: | ---: | ---: |
| 349,440 | 0.981 | 1.015 | 1.044 |
| 2,097,152 | 1.098 | 0.991 | 1.007 |

These results are not a workload-speedup claim: the gain is not robust because Neg is Amdahl-limited inside the unfused focal-loss chain. The control harness is retained to make this boundary reproducible. A separate fused focal-loss proposal may target the remaining passes; it is intentionally not bundled into this kernel PR.

## Reproduction commands

```sh
go test -tags goai_bench_control ./backend/cpu -run '^$' -bench '^BenchmarkNegF32Paired/' -benchtime=100x -count=7 -benchmem
go test ./backend/metal -run '^$' -bench '^BenchmarkMetalNegRouteExtension/' -benchtime=100x -count=7 -benchmem
GOEXPERIMENT=simd go test ./backend/metal -run '^$' -bench '^BenchmarkMetalNegRouteExtension/' -benchtime=100x -count=7 -benchmem
go test ./backend/metal -run '^$' -bench '^BenchmarkMetalUnaryRouteCandidates/neg/' -benchtime=100x -count=7 -benchmem
GOEXPERIMENT=simd go test ./backend/metal -run '^$' -bench '^BenchmarkMetalUnaryRouteCandidates/neg/' -benchtime=100x -count=7 -benchmem
go test -tags goai_bench_control ./nn -run '^$' -bench '^BenchmarkSigmoidFocalNegRoute/' -benchtime=20x -count=7 -benchmem
```

Metal commands require unsandboxed GPU access. Benchmark-control code is excluded from production binaries unless `goai_bench_control` is explicitly enabled.

## Validation

- package-wide CPU, Metal, and NN tests pass in the default build;
- package-wide CPU and Metal tests pass with `GOEXPERIMENT=simd`;
- tagged control CPU and NN suites pass;
- focused exactness and loss parity pass under the race detector;
- portable fallback tests cross-compile for linux/amd64;
- repository `make preflight` passes;
- focused in-tree and direct external perfscan scans report no candidate in the new Neg files;
- `git diff --check` passes;
- Spectackle typed reindex: 2,554 files, 17,557 nodes, 32,631 edges, 6,714 typed calls, zero skipped;
- Spectackle check reports no errors or tranche-specific warnings after guard-region hardening.

The full SIMD NN package reaches an untouched deterministic `TestKANForwardIsBitIdentical` digest mismatch. The exact base commit reproduces the same received and expected digests byte-for-byte, so it is recorded as a pre-existing failure rather than attributed to this tranche.
