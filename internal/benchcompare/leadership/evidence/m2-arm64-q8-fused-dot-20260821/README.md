# M2 ARM64 Q8_0 fused decode-dot evidence — 2026-08-21

GoAI's ARM64 single-token Q8_0 path now widens signed int8 quants, applies
each f16 block scale, multiplies contiguous f32 activations, and reduces the
entire row in one NEON call. It does not materialize dequantized weights. The
portable scalar path, every non-ARM64 build, every M>1 path, and all other
quant formats are unchanged.

## Result

| Cell | scalar control | ARM64 NEON | change | speedup |
|---|---:|---:|---:|---:|
| Q8_0 row dot, K=4096 | 4,849.5 ns | 326.8 ns | -93.26% | 14.84x |
| QMatMul M1, N=64, K=1024 | 22.362 us | 5.470 us | -75.54% | 4.09x |
| QMatMul M1, N=4096, K=1024 | 258.7 us | 103.5 us | -59.99% | 2.50x |
| Quantized Mamba2 recurrent decode, Q8_0 | 169.60 us | 87.98 us | -48.13% | 1.93x |
| Quantized Mamba2 recurrent decode, Q6_K | 100.1 us | 100.2 us | statistically flat | 1.00x |

Every Q8_0 time comparison has p=0.000 and n=10. The Q6_K negative
control has p=0.579. Allocation counts are unchanged: zero at the leaf, 4
and 21 in the QMatMul cells, and 93 in recurrent decode.

The leaf and QMatMul B/s values count logical f32 work, not encoded Q8_0
bytes. They are retained only because this is the benchmark harness's
established convention; they are not physical memory-bandwidth claims.

## Protocol

- Hardware: Apple M2 Pro, darwin/arm64.
- OS: macOS 26.5.1 (25F80).
- Toolchain: Go 1.26.6; benchstat from golang.org/x/perf.
- Base: `0d8ffffc3bad4ab2314d62200c93afd8cb532842` (main after PR #1133).
- Branch: `codex/m2-arm64-q8-dot`.
- The leaf and QMatMul controls and candidates run in one binary. Odd
  invocations set `GOAI_Q8_NEON_FIRST=1` to reverse sub-benchmark order.
- Mamba2 uses separately precompiled base and candidate binaries; execution
  order reverses on every invocation.
- Every cell ran eleven 250 ms samples. The first sample of every benchmark
  was discarded before benchstat, leaving n=10. No other sample was removed.
- Q6_K ran beside Q8_0 as an untouched end-to-end negative control.
- Tests were selected by invoking an already-compiled test binary with
  `-test.run`; `go test -run` was not used.
- External perfscan is `github.com/jxsl13/perfscan/perfscan@v1.71.0` and was
  run with `GOPROXY=direct`. The new Q8 files produce no findings; the full
  repository still has pre-existing findings and config-version warnings.
- The generalizable cross-architecture selector and whole-row call-amortization
  finding is recorded at perfscan issue #799, comment 5372812931.

## Numerical gate

The assembly preserves Q8_0 element mapping and the exact f16-table scale,
but uses four f32 vector accumulators instead of the portable per-element f64
sum. It therefore makes a bounded numerical-equivalence claim, not a bitwise
summation-order claim.

Three gates cover the contract:

1. One block with scale=1, all quants=1, and all activations=1 returns exactly 32.
2. One hundred arbitrary raw rows spanning K=32 through K=4096 have maximum
   scalar-relative error 2.93272224e-5 and verify input immutability.
3. The full fused-vs-general QMatMul matrix passes its architecture tolerance.

The observed maximum is below the 1e-4 contract.

## Claim boundary

This is an internal Apple ARM64 leadership gain. It closes the missing Q8_0
SIMD-selector defect and demonstrates end-to-end retention in recurrent
decode. It does not by itself close the historical 8.8x Q8_0-vs-f32
whole-model cell or claim superiority over llama.cpp. Those require rerunning
the original model and a matched incumbent comparison with identical quant
bytes, activations, threads, shapes, transfers, and measurement boundaries.
