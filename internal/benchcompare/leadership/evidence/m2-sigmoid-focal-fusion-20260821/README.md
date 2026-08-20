# M2 Sigmoid Focal core and VJP fusion — 2026-08-21

## Claim boundary

This evidence establishes a measured GoAI winner matrix against GoAI's frozen pre-fusion composite implementation on one Apple M2 Pro. It does not claim universal cross-library leadership. Each pair uses identical tensors, dtype, gamma, alpha, reduction, warmup, allocation, tape, and synchronization boundaries in the same binary.

The retained change fuses the per-element stable sigmoid focal term and its logits VJP. `OpMean` remains explicit so reduction order is unchanged. The production route is capability-gated; unsupported backends, mixed dtypes, and strided inputs retain the original composite graph without an implicit fused-op migration.

## Pinned environment

| Item | Value |
| --- | --- |
| Hardware | Apple M2 Pro, darwin/arm64 |
| OS | macOS 26.5.1 (25F80) |
| Go | go1.26.6 darwin/arm64 |
| Spectackle | v0.9.3, built with Go 1.26.6 |
| Base | `959a8a2309e16678e5a67c20d5647fa95ca16b76` |
| Implementation checkpoint | `c22df4e51f0f075ea9e2042e8097a3f562a281dd` |
| Workload | F32 logits/0-or-1 targets, gamma=2, alpha=0.25, global mean |
| Default protocol | 3 untimed warmups per arm, 10 iterations, count=7, per-cell median, 3 independent campaigns |
| SIMD protocol | `GOEXPERIMENT=simd`, 3 untimed warmups per arm, 10 iterations, count=7, per-cell median |

Spectackle proposal `P-01M0GK31T4FKN`, task `T-01M0GK57K7FF3`, and rules `SIGMOID-FOCAL-FUSION-001`, `SIGMOID-FOCAL-FUSION-002`, and `SIGMOID-FOCAL-FUSION-PERF-001` govern the change.

## Architecture and semantic contract

The change stays within the existing architecture:

- ADR-0003: append-only `OpSigmoidFocalCore` and `OpSigmoidFocalCoreBackward` kernel registration through `Execute`;
- ADR-0006: one registered VJP dispatches the fused backward op and returns no target gradient, preserving detached-label semantics;
- ADR-0014: `SigmoidFocalAttrs` carries gamma and alpha through execution and the tape;
- ADR-0028: the opt-in M2 SIMD path composes on the shared f32-native exp/log/sigmoid leaf and its ADR-0021 tolerance.

The core emits the per-element weighted term

`alpha_t * softplus(z) * exp(-gamma * softplus(-z))`, where `z=(1-2y)*x`.

For gamma not equal to zero, both softplus values share one exact stable base: `b=log1p(exp(-abs(z)))`, `softplus(z)=b+max(z,0)`, and `softplus(-z)=b+max(-z,0)`. This removes one exponential and one logarithm per element. The default F32 implementation retains `b` in F64 for the add and narrows each softplus result separately, preserving the old graph's rounding barriers. The default F32/F64 forward and VJP are bit-exact against the composite oracle. `GOEXPERIMENT=simd` F32 is checked within ADR-0021's `1e-6 + 2e-3*abs(reference)` envelope; F64 remains exact.

General analyzer findings: [shared complementary softplus work, perfscan #781](https://github.com/jxsl13/perfscan/issues/781), and [approximate-exp clamp amplification in fused VJPs, perfscan #782](https://github.com/jxsl13/perfscan/issues/782).

## Default M2 production route

Speedup is frozen composite median divided by fused production median. Every one of the 12 independent campaign cells exceeds the 1.10 promotion gate; the weakest is 1.501x.

| Elements | Mode | C1 control/candidate | C1 x | C2 control/candidate | C2 x | C3 control/candidate | C3 x |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 349,440 | forward | 4.875 / 2.697 ms | 1.807 | 4.908 / 2.717 ms | 1.807 | 4.861 / 2.708 ms | 1.795 |
| 349,440 | forward+backward | 14.178 / 6.353 ms | 2.232 | 13.968 / 6.282 ms | 2.224 | 13.964 / 6.310 ms | 2.213 |
| 2,097,152 | forward | 23.246 / 13.347 ms | 1.742 | 23.336 / 15.546 ms | 1.501 | 22.994 / 13.715 ms | 1.677 |
| 2,097,152 | forward+backward | 81.311 / 31.854 ms | 2.553 | 78.668 / 31.147 ms | 2.526 | 78.815 / 34.715 ms | 2.270 |

The allocation boundary improves independently of timing variance:

| Elements | Mode | Control bytes/op | Candidate bytes/op | Control allocs/op | Candidate allocs/op |
| ---: | --- | ---: | ---: | ---: | ---: |
| 349,440 | forward | 16.81 MB | 1.40 MB | 78–80 | 14 |
| 349,440 | forward+backward | 40.65 MB | 4.20 MB | 214 | 40 |
| 2,097,152 | forward | 100.67 MB | 8.39 MB | 80–81 | 14 |
| 2,097,152 | forward+backward | 243.30 MB | 25.17 MB | 218–219 | 40 |

## M2 SIMD route

The real opt-in NEON configuration also wins in every count-7 cell. This remains a secondary compatibility/winner-zone check rather than part of the default-build promotion rule. A cold exact fallback below the approximate exponential's underflow clamp preserves extreme-tail VJP semantics without removing the normal SIMD route.

| Elements | Mode | Control | Candidate | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 349,440 | forward | 4.644 ms | 2.149 ms | 2.161 |
| 349,440 | forward+backward | 6.777 ms | 5.531 ms | 1.225 |
| 2,097,152 | forward | 21.271 ms | 11.373 ms | 1.870 |
| 2,097,152 | forward+backward | 35.074 ms | 27.510 ms | 1.275 |

## Reproduction commands

```sh
go test -tags goai_bench_control ./nn -run '^$' -bench '^BenchmarkSigmoidFocalFusion/n(349440|2097152)/(forward|forward_backward)/(control|candidate)$' -benchtime=10x -count=7 -benchmem
GOEXPERIMENT=simd go test -tags goai_bench_control ./nn -run '^$' -bench '^BenchmarkSigmoidFocalFusion/n(349440|2097152)/(forward|forward_backward)/(control|candidate)$' -benchtime=10x -count=7 -benchmem
```

Benchmark-control code is excluded from production binaries unless `goai_bench_control` is explicitly enabled.

## Validation

- direct CPU/reference forward and backward parity covers F32/F64, gamma zero/nonzero, empty, short, parallel-sized, signed-zero, subnormal, extreme-finite, Inf, and NaN tensors;
- fused-vs-composite loss and VJP parity covers F32/F64 and verifies that targets remain detached;
- finite-difference gradcheck covers the fused logits VJP;
- production route tests prove two recorded ops for fusion and nine for the gamma-nonzero fallback;
- mixed-dtype and strided inputs prove the composite guard path;
- focused default and `GOEXPERIMENT=simd` suites pass, including explicit NaN-class and Inf-sign assertions;
- `git diff --check` passes;
- `make preflight`, focused race, Linux/amd64 cross-build, full SIMD cross-build, and `make preflight-metal` pass;
- external perfscan v1.71.0 reports no findings in the changed production files;
- Spectackle check has no errors or tranche-specific warnings; reindex records 2,563 files, 17,608 nodes, and 32,723 edges, with the typed-call pass unavailable because `go/packages` reports invalid package names, so this checkpoint has a syntactic graph only.
