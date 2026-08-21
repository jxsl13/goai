# M2 ARM64 Q3_K fused decode-dot evidence — 2026-08-21

GoAI's ARM64 single-token Q3_K path now reconstructs the two-bit plane and
inverted high-mask bit, applies all sixteen signed sub-block scales, and
reduces each 256-weight superblock against contiguous f32 activations in one
NEON call. It does not materialize dequantized weights. The portable scalar
path, every non-ARM64 build, every M>1 path, and all other quant formats are
unchanged.

## Result

| Cell | scalar control | ARM64 NEON | change | speedup |
|---|---:|---:|---:|---:|
| Q3_K row dot, K=4096 | 5,027.0 ns | 735.4 ns | -85.37% | 6.84x |
| QMatMul M1, N=64, K=1024 | 81.40 us | 12.40 us | -84.76% | 6.56x |
| QMatMul M1, N=4096, K=1024 | 811.4 us | 180.0 us | -77.81% | 4.51x |
| Quantized Mamba2 recurrent decode, Q3_K | 392.3 us | 128.7 us | -67.21% | 3.05x |
| Quantized Mamba2 recurrent decode, Q5_K | 124.7 us | 124.4 us | statistically flat | 1.00x |

Every Q3_K time comparison has p=0.000 and n=10. The Q5_K negative control
has p=0.247. Allocation counts are unchanged: zero at the leaf, 4 and 29 in
the QMatMul cells, and 93 in recurrent decode.

The QMatMul B/s values count logical f32 work, not encoded Q3_K bytes. They
are retained only because this is the benchmark harness's established
convention; they are not physical memory-bandwidth claims.

## Protocol

- Hardware: Apple M2 Pro, darwin/arm64.
- OS: macOS 26.5.1 (25F80).
- Toolchain: Go 1.26.6; benchstat from golang.org/x/perf.
- Base: `c856022516709dbdb2c09906d5784c3084833a77` (main after PR #1135).
- Branch: `codex/m2-arm64-q3k-dot`.
- Leaf controls and candidates run from one binary as separate exact
  benchmarks. Their execution order reverses on every retained pair.
- QMatMul controls and candidates run in one binary through the production
  selector. Alternate invocations set `GOAI_GGUF_Q3K_NEON_FIRST=1` to
  reverse sub-benchmark order.
- Mamba2 uses separately precompiled base and candidate binaries; execution
  order reverses on every invocation.
- Each cell had an excluded warm-up in every exercised order followed by ten
  retained 250 ms samples. No retained sample was removed.
- Q5_K ran beside Q3_K as an untouched end-to-end negative control.
- Tests were selected by invoking an already-compiled test binary with
  `-test.run`; `go test -run` was not used.
- External perfscan is `github.com/jxsl13/perfscan/perfscan@v1.71.0` and was
  run repository-wide with `GOPROXY=direct`. The new Q3_K implementation
  files produce no findings; the repository has pre-existing findings.
- The generalizable architecture-selector asymmetry is reported on perfscan
  issue #799, including the Q3_K confirmation comment.

## Numerical gate

The assembly preserves Q3_K element mapping, the inverted high-mask rule,
and signed scale selection, but uses four f32 vector accumulators instead of
the portable per-element f64 sum. It therefore makes a bounded
numerical-equivalence claim, not a bitwise summation-order claim.

Three gates cover the contract:

1. One synthetic block with all low streams `0,1,2,3`, half-set high masks,
   unit coefficients, and unit activations returns exactly -128.
2. One hundred arbitrary raw rows spanning K=256 through K=4096 have maximum
   scalar-relative error 1.5923603989341695e-5 and verify input immutability.
3. The full fused-vs-general QMatMul matrix passes its architecture tolerance.

The observed maximum is more than 6x below the 1e-4 contract.

## Validation

The complete GGUF suite and its race build pass. Linux amd64 portable,
Linux amd64 `simd`, and Linux ARM64 test binaries cross-compile. The complete
NLP suite passes when the known `TestDiffusionLMGrammarE2E` baseline failure
is skipped; the immutable base and candidate binaries produce the identical
CE, generated string, and failure for that test. Objdump confirms the selected
leaf contains vector table lookup, signed conversion, multiply/FMA, and pairwise
reduction instructions.

## Claim boundary

This is an internal Apple ARM64 leadership gain. It closes another scalar
K-quant decode selector edge and demonstrates end-to-end retention in
recurrent decode. It does not claim superiority over llama.cpp. A matched
incumbent comparison still requires identical quant bytes, activations,
threads, shapes, transfers, and measurement boundaries.
