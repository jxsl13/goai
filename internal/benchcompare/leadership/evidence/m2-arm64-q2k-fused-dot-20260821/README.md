# M2 ARM64 Q2_K fused decode-dot evidence — 2026-08-21

GoAI's ARM64 single-token Q2_K path now unpacks two-bit weights, applies all
sixteen affine scale/min pairs, and reduces each 256-weight superblock against
contiguous f32 activations in one NEON call. It does not materialize
dequantized weights. The portable scalar path, every non-ARM64 build, every
M>1 path, and all other quant formats are unchanged.

## Result

| Cell | scalar control | ARM64 NEON | change | speedup |
|---|---:|---:|---:|---:|
| Q2_K row dot, K=4096 | 4,849 ns | 1,066 ns | -78.02% | 4.55x |
| QMatMul M1, N=64, K=1024 | 77.17 us | 17.02 us | -77.94% | 4.53x |
| QMatMul M1, N=4096, K=1024 | 750.8 us | 227.4 us | -69.71% | 3.30x |
| Quantized Mamba2 recurrent decode, Q2_K | 374.2 us | 147.9 us | -60.46% | 2.53x |
| Quantized Mamba2 recurrent decode, Q5_K | 124.3 us | 123.9 us | statistically flat | 1.00x |

Every Q2_K time comparison has p=0.000 and n=10. The untouched Q5_K negative
control has p=0.247. Allocation counts are unchanged: zero at the leaf, 4 and
29 in the QMatMul cells, and 93 in recurrent decode. No retained sample was
removed, including visible slow outliers in the scalar leaf, candidate N=4096
path, and candidate Q5_K control.

The QMatMul B/s values count logical f32 work, not encoded Q2_K bytes. They
are retained only because this is the benchmark harness's established
convention; they are not physical memory-bandwidth claims.

## Protocol

- Hardware: Apple M2 Pro, darwin/arm64.
- OS: macOS 26.5.1 (25F80).
- Toolchain: Go 1.26.6; benchstat from golang.org/x/perf.
- Base: 512d496a2fd66d40bfc0c31b264396524f2b3e05 (main after PR #1136).
- Branch: codex/m2-arm64-q2k-dot.
- Leaf controls and candidates run from one binary as separate exact
  benchmarks. Their execution order reverses on every retained pair.
- QMatMul controls and candidates run in one binary through the production
  selector. Alternate invocations set GOAI_GGUF_Q2K_NEON_FIRST=1 to reverse
  sub-benchmark order.
- Mamba2 uses separately precompiled base and candidate binaries; execution
  order reverses on every invocation.
- Each exercised order had an excluded warm-up followed by ten retained
  250 ms samples. No retained sample was removed.
- Q5_K ran beside Q2_K as an untouched end-to-end negative control.
- Tests were selected by invoking already-compiled test binaries with
  -test.run; go test -run was not used.
- External perfscan is github.com/jxsl13/perfscan/perfscan@v1.71.0 and is
  resolved with GOPROXY=direct.
- The generalized selector-family and cancellation-safe accumulation finding
  is reported on perfscan issue #799.

## Numerical gate

The assembly preserves Q2_K element mapping and the f32 operation
step*q-offset, then widens dequantized weights and activations before
eight-bank f64 accumulation. The widening was required by a real
near-cancellation failure from an f32 pilot; the 1e-4 contract was not relaxed.

Three gates cover the contract:

1. One synthetic block with all packed streams 0,1,2,3, sequential steps
   1 through 16, unit offsets, and unit activations returns exactly 3648.
2. One hundred arbitrary raw rows spanning K=256 through K=4096 have observed
   maximum scalar-relative error 0 and verify input immutability.
3. The full fused-vs-general QMatMul matrix passes its architecture tolerance.

The selected leaf is allocation-free. Objdump confirms table expansion,
unsigned conversion, affine f32 multiply/subtract, f32-to-f64 widening, f64
FMA, and pairwise reduction in the production binary.

## Claim boundary

This is an internal Apple ARM64 leadership gain. It closes the final scalar
K-quant selector edge and demonstrates end-to-end retention in recurrent
decode. It does not claim superiority over llama.cpp. A matched incumbent
comparison still requires identical quant bytes, activations, threads, shapes,
transfers, and measurement boundaries.

