# M2 ARM64 Q5_K fused decode-dot evidence — 2026-08-21

GoAI's ARM64 single-token Q5_K path now inserts the fifth-bit plane into the
packed nibbles, applies the affine scale/min coefficients, and reduces each
256-weight superblock against contiguous f32 activations in one NEON call. It
does not materialize dequantized weights. The portable scalar path, every
non-ARM64 build, every M>1 path, and all other quant formats are unchanged.

## Result

| Cell | scalar control | ARM64 NEON | change | speedup |
|---|---:|---:|---:|---:|
| Q5_K row dot, K=4096 | 4,824.5 ns | 666.4 ns | -86.19% | 7.24x |
| QMatMul M1, N=64, K=1024 | 76.50 us | 10.95 us | -85.69% | 6.99x |
| QMatMul M1, N=4096, K=1024 | 735.4 us | 162.3 us | -77.93% | 4.53x |
| Quantized Mamba2 recurrent decode, Q5_K | 364.3 us | 124.5 us | -65.83% | 2.93x |
| Quantized Mamba2 recurrent decode, Q6_K | 100.4 us | 100.5 us | statistically flat | 1.00x |

Every Q5_K time comparison has p=0.000 and n=10. The Q6_K negative control
has p=0.684. Allocation counts are unchanged: zero at the leaf, 4 and 29 in
the QMatMul cells, and 93 in recurrent decode.

The QMatMul B/s values count logical f32 work, not encoded Q5_K bytes. They
are retained only because this is the benchmark harness's established
convention; they are not physical memory-bandwidth claims.

## Protocol

- Hardware: Apple M2 Pro, darwin/arm64.
- OS: macOS 26.5.1 (25F80).
- Toolchain: Go 1.26.6; benchstat from golang.org/x/perf.
- Base: `4aba06d12d0ba35df9cc5b8493076d618fbb2f17` (main after PR #1134).
- Branch: `codex/m2-arm64-q5k-dot`.
- Leaf controls and candidates run from one binary as separate exact
  benchmarks. Their execution order reverses on every retained pair.
- QMatMul controls and candidates run in one binary through the production
  selector. Alternate invocations set `GOAI_GGUF_Q5K_NEON_FIRST=1` to
  reverse sub-benchmark order.
- Mamba2 uses separately precompiled base and candidate binaries; execution
  order reverses on every invocation.
- Each cell had an excluded warm-up in every exercised order followed by ten
  retained 250 ms samples. No retained sample was removed.
- Q6_K ran beside Q5_K as an untouched end-to-end negative control.
- Tests were selected by invoking an already-compiled test binary with
  `-test.run`; `go test -run` was not used.
- External perfscan is `github.com/jxsl13/perfscan/perfscan@v1.71.0` and was
  run with `GOPROXY=direct`. The new Q5_K files produce no findings; the
  package still has pre-existing findings.
- The generalizable architecture-selector asymmetry is reported on perfscan
  issue #799.

## Numerical gate

The assembly preserves Q5_K element mapping, scale/min selection, and the
operation order `step*q-offset`, but uses four f32 vector accumulators instead
of the portable per-element f64 sum. It therefore makes a bounded
numerical-equivalence claim, not a bitwise summation-order claim.

Three gates cover the contract:

1. One block with all qh bits set, packed nibbles `0x21`, unit coefficients,
   and unit activations returns exactly 4,480.
2. One hundred arbitrary raw rows spanning K=256 through K=4096 have maximum
   scalar-relative error 9.359696090657784e-6 and verify input immutability.
3. The full fused-vs-general QMatMul matrix passes its architecture tolerance.

The observed maximum is more than 10x below the 1e-4 contract.

## Claim boundary

This is an internal Apple ARM64 leadership gain. It closes the missing Q5_K
SIMD-selector defect and demonstrates end-to-end retention in recurrent
decode. It does not claim superiority over llama.cpp. A matched incumbent
comparison still requires identical quant bytes, activations, threads,
shapes, transfers, and measurement boundaries.
